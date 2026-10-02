"""Transactional low-confidence and fabrication case persistence."""
from datetime import datetime
from sqlalchemy import select, func
from app.db.models import Conversation, LowConfidenceQuestion, FaithCase


class QualityLedger:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def add_low_confidence(self, conversation_key, raw_question, source, reason):
        if source not in {'retrieval_low_conf', 'self_check', 'user_feedback'} or not raw_question.strip():
            raise ValueError('invalid low-confidence entry')
        with self.session_factory.begin() as s:
            conv = s.get(Conversation, conversation_key)
            if conv is None or conv.id is None:
                raise ValueError('unknown conversation')
            row = LowConfidenceQuestion(conversation_id=conv.id, raw_question=raw_question, source=source, reason=reason)
            s.add(row)
            s.flush()
            return row.id

    def record_faith_case(self, case):
        allowed = {'eval_id','bucket','query','strategy','answer','reason','citations','judge_model'}
        values = {key: case[key] for key in allowed if key in case}
        with self.session_factory.begin() as s:
            if s.bind.dialect.name == 'mysql':
                from sqlalchemy.dialects.mysql import insert
                stmt = insert(FaithCase).values(**values)
                s.execute(stmt.on_duplicate_key_update(**{k:v for k,v in values.items() if k != 'eval_id'},
                    seen_count=FaithCase.seen_count+1, last_seen_at=func.now(), status='未解决', resolution=None))
                return s.scalar(select(FaithCase.id).where(FaithCase.eval_id == values['eval_id']))
            row = s.scalar(select(FaithCase).where(FaithCase.eval_id == values['eval_id']))
            if row is None:
                row = FaithCase(**values)
                s.add(row)
            else:
                for key, value in values.items():
                    setattr(row, key, value)
                row.seen_count += 1
                row.last_seen_at = datetime.now()
                row.status = '未解决'
                row.resolution = None
            s.flush()
            return row.id

    def resolve(self, eval_id, status, resolution):
        if status not in {'未解决', '已解决', '无需解决'}:
            raise ValueError('invalid disposition')
        explanation = (resolution or '').strip()
        if status != '未解决' and not 1 <= len(explanation) <= 300:
            raise ValueError('resolution required, at most 300 characters')
        with self.session_factory.begin() as s:
            row = s.scalar(select(FaithCase).where(FaithCase.eval_id == eval_id).with_for_update())
            if row is None:
                raise LookupError(eval_id)
            row.status = status
            row.resolution = explanation if status != '未解决' else None
            if status != '未解决':
                row.resolved_at = datetime.now()

    def list_cases(self, status=None):
        with self.session_factory() as s:
            query = select(FaithCase).order_by(FaithCase.last_seen_at.desc(), FaithCase.id)
            if status:
                query = query.where(FaithCase.status == status)
            return list(s.scalars(query))
