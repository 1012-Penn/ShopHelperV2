"""The single execution boundary: validation, permissions, retries and audit."""

import asyncio
import inspect
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict

import httpx
from jsonschema import validators
from sqlalchemy.exc import IntegrityError

from app.db.models import ToolWriteReceipt

from .definitions import ExecutionContext, ToolResult
from .formatting import decode, format_result
from .policy import ToolPolicy


class ToolEngine:
    def __init__(
        self,
        registry,
        timeout_seconds=5,
        max_retries=2,
        policy=None,
        audit=None,
        session_factory=None,
    ):
        self.registry, self.timeout_seconds, self.max_retries = (
            registry,
            timeout_seconds,
            max_retries,
        )
        self.policy, self.audit, self.session_factory = (
            policy or ToolPolicy(),
            audit,
            session_factory,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="mewhelp-tool"
        )
        self._memory_receipts = {}
        from threading import Lock

        self._receipt_lock = Lock()

    def preflight(self, name, args, tool_call_id, context, require_confirmation=False):
        _, _, result = self._prepare(
            name, args, tool_call_id, context, require_confirmation
        )
        return result

    def _prepare(self, name, args, tool_call_id, context, require_confirmation):
        """Resolve once and authorize the exact executor used by this invocation."""
        try:
            definition = self.registry.get(name)
        except ValueError:
            return (
                None,
                "deny",
                ToolResult(
                    name,
                    tool_call_id,
                    "未注册或未授权的工具，不能执行。",
                    True,
                    "permission_denied",
                    "permission_denied",
                    source="unknown",
                ),
            )
        validator = validators.validator_for(definition.input_schema)(
            definition.input_schema
        )
        errors = list(validator.iter_errors(args))
        if errors:
            details = [
                {
                    "field": ".".join(map(str, e.absolute_path)) or "$",
                    "message": e.message,
                }
                for e in errors
            ]
            return (
                definition,
                "deny",
                ToolResult(
                    name,
                    tool_call_id,
                    json.dumps(
                        {"error": "参数不合法", "details": details}, ensure_ascii=False
                    ),
                    True,
                    "validation_blocked",
                    "invalid_arguments",
                    source=definition.source,
                ),
            )
        permission = self.policy.permission(definition.source, name)
        denied = permission == "deny" or (
            permission == "write"
            and (
                not context.ticket_requested
                or (require_confirmation and context.confirmed_call_id != tool_call_id)
            )
        )
        if denied:
            return (
                definition,
                permission,
                ToolResult(
                    name,
                    tool_call_id,
                    "权限拒绝：写操作需要用户明确要求，并确认本次预览；外部工具需要本地授权。",
                    True,
                    "permission_denied",
                    "permission_denied",
                    source=definition.source,
                ),
            )
        return definition, permission, None

    def _reserve(self, context, tool_call_id):
        key = (context.conversation_id, tool_call_id)
        if not self.session_factory:
            with self._receipt_lock:
                if key in self._memory_receipts:
                    return False
                self._memory_receipts[key] = True
                return True
        try:
            with self.session_factory.begin() as session:
                session.add(
                    ToolWriteReceipt(
                        conversation_id=key[0], tool_call_id=key[1], status="running"
                    )
                )
            return True
        except IntegrityError:
            return False

    def _save_receipt(self, context, result):
        if not self.session_factory:
            return
        try:
            with self.session_factory.begin() as session:
                receipt = session.get(
                    ToolWriteReceipt, (context.conversation_id, result.tool_call_id)
                )
                receipt.status = "created" if not result.is_error else "unknown"
                receipt.result = asdict(result)
        except Exception:
            logging.getLogger(__name__).exception(
                "Write receipt outcome unknown; reservation remains"
            )

    def _invoke(self, definition, args, context):
        value = definition.handler(args, context)
        if inspect.isawaitable(value):

            async def bounded():
                return await asyncio.wait_for(value, timeout=self.timeout_seconds)

            return asyncio.run(bounded())
        return value

    @staticmethod
    def _transient(error):
        if isinstance(
            error,
            (
                TimeoutError,
                FutureTimeoutError,
                asyncio.TimeoutError,
                ConnectionError,
                httpx.TransportError,
            ),
        ):
            return True
        nested = getattr(error, "exceptions", ())
        return bool(nested) and all(ToolEngine._transient(e) for e in nested)

    def run(self, name, args, tool_call_id, context=None):
        context = context or ExecutionContext()
        start = time.monotonic()
        # Keep the validated definition and permission together. Discovery may
        # replace this name concurrently, but cannot swap in another executor.
        definition, permission, result = self._prepare(
            name, args, tool_call_id, context, True
        )
        if result is None:
            write = permission == "write"
            if write and not self._reserve(context, tool_call_id):
                return ToolResult(
                    name,
                    tool_call_id,
                    "本次工单已提交或结果暂未确认，请勿重复提交。",
                    True,
                    "permission_denied",
                    "permission_denied",
                    source=definition.source,
                )
            attempts = 0 if write else self.max_retries
            for attempt in range(attempts + 1):
                future = self._executor.submit(self._invoke, definition, args, context)
                try:
                    value = future.result(timeout=self.timeout_seconds)
                    data = decode(value)
                    empty = isinstance(data, dict) and any(
                        data.get(k) is False for k in ("found", "matched")
                    )
                    result = ToolResult(
                        name,
                        tool_call_id,
                        format_result(definition, value),
                        empty,
                        "failed" if empty else "success",
                        "not_found" if empty else None,
                        attempt,
                        source=definition.source,
                    )
                    break
                except Exception as error:  # noqa: BLE001 - isolate arbitrary tool handlers
                    future.cancel()
                    timeout = isinstance(
                        error, (TimeoutError, FutureTimeoutError, asyncio.TimeoutError)
                    )
                    if attempt < attempts and self._transient(error):
                        continue
                    message = (
                        "工单提交结果暂未确认，请勿重复提交；请联系人工客服核查。"
                        if write
                        else (
                            "工具执行超时，请稍后再试。"
                            if timeout
                            else "工具执行故障，请稍后再试。"
                        )
                    )
                    result = ToolResult(
                        name,
                        tool_call_id,
                        message,
                        True,
                        "timeout" if timeout else "failed",
                        "execution_error",
                        attempt,
                        source=definition.source,
                        error_detail=type(error).__name__ + ": " + str(error)[:1000],
                    )
                    break
            if write:
                self._save_receipt(context, result)
        result = ToolResult(
            **{**asdict(result), "duration_ms": (time.monotonic() - start) * 1000}
        )
        if self.audit:
            try:
                self.audit.record(result, args, context)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Tool audit failed, preserving outcome"
                )
        return result

    def reject(self, name, args, tool_call_id, context, message):
        try:
            source = self.registry.get(name).source
        except ValueError:
            source = "unknown"
        result = ToolResult(
            name,
            tool_call_id,
            message,
            True,
            "permission_denied",
            "permission_denied",
            source=source,
        )
        if self.audit:
            try:
                self.audit.record(result, args, context)
            except Exception:
                logging.getLogger(__name__).exception("Audit failed")
        return result

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
