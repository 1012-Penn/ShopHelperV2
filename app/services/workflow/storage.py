"""Business message persistence and per-conversation request serialization."""
from contextlib import contextmanager
from threading import Lock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import select

from app.db.models import Conversation, Message


class ConversationLocks:
    def __init__(self):
        self._guard = Lock()
        self._entries = {}

    @contextmanager
    def hold(self, conversation_id):
        with self._guard:
            entry = self._entries.setdefault(conversation_id, [Lock(), 0])
            entry[1] += 1
        entry[0].acquire()
        try:
            yield
        finally:
            entry[0].release()
            with self._guard:
                entry[1] -= 1
                if not entry[1]:
                    del self._entries[conversation_id]


class ConversationStore:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def check_owner(self, conversation_id, user_id):
        with self.session_factory() as s:
            row = s.get(Conversation, conversation_id)
            if row is not None and row.user_id != user_id:
                raise ValueError('conversation does not belong to user')

    def prepare(self, request):
        with self.session_factory.begin() as s:
            conv = s.get(Conversation, request.conversation_id)
            if conv is not None and conv.user_id != request.user_id:
                raise ValueError('conversation does not belong to user')
            if conv is None:
                s.add(Conversation(conversation_id=request.conversation_id, user_id=request.user_id, status='open'))
                s.flush()
            rows = list(s.scalars(select(Message).where(
                Message.conversation_id == request.conversation_id).order_by(Message.id)))
            s.add(Message(conversation_id=request.conversation_id, role='user', content=request.message))
        return [self.to_message(row) for row in rows]

    @staticmethod
    def to_message(row):
        values = {'content': row.content, 'id': f'sql-{row.id}'}
        if row.role == 'user':
            return HumanMessage(**values)
        if row.role == 'tool':
            return ToolMessage(**values, tool_call_id=row.tool_call_id or 'missing-id')
        return AIMessage(**values, tool_calls=row.tool_calls or [])

    def save_answer(self, conversation_id, answer, citations, actions, question):
        metadata = {'items': actions, 'question': question,
                    'ticket': {'status': 'offered'}} if actions else None
        with self.session_factory.begin() as s:
            row = Message(conversation_id=conversation_id, role='assistant', content=answer,
                          citations=citations or None, actions=metadata)
            s.add(row)
            s.flush()
            return row.id

    def save_tool_pair(self, conversation_id, message, observations):
        with self.session_factory.begin() as s:
            s.add(Message(conversation_id=conversation_id, role='assistant', content='',
                          tool_calls=message.tool_calls))
            for observation in observations:
                s.add(Message(conversation_id=conversation_id, role='tool', content=observation.content,
                              tool_call_id=observation.tool_call_id))
