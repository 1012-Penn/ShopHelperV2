"""Privacy scrubbing before customer dialogue is sent to an extraction model."""

import re

_PATTERNS = (
    (re.compile(r"(?<!\d)(?:(?:\+|00)?86[-\s]?)?1[3-9]\d{9}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![A-Z0-9-])"), "[EMAIL]"),
    (re.compile(r"(?<!\d)[1-9]\d{16}[0-9Xx](?!\w)"), "[ID_CARD]"),
    (
        re.compile(
            r"(?i)(?:订单(?:编号|号)|工单(?:编号|号)|order\s*(?:id|no)?|ticket\s*(?:id|no)?)"
            r"\s*(?:为|是|[:：#])?\s*[A-Z0-9][A-Z0-9_-]{2,}"
        ),
        "[REDACTED_ID]",
    ),
    (
        re.compile(r"(?i)(?:ORD|ORDER|TKT|TICKET|CASE)[-_]?[A-Z0-9][A-Z0-9_-]{2,}(?![A-Z0-9_-])"),
        "[REDACTED_ID]",
    ),
)


def redact_customer_data(text: str) -> str:
    redacted = text
    for pattern, replacement in _PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted
