"""Parse WeChat chat records from various formats into structured conversations."""

import re
from dataclasses import dataclass


@dataclass
class ChatLine:
    sender: str
    time: str
    content: str


@dataclass
class ParsedChat:
    lines: list[ChatLine]
    raw_text: str
    is_chat_record: bool  # whether this looks like a chat record


# Common patterns for WeChat chat records (copy-paste format)
# Pattern: "name time\nmessage" or "name：message"
_PATTERN_FULL = re.compile(
    r"^(?P<name>.{1,20}?)\s+"
    r"(?P<time>\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)"
    r"\s*$",
    re.MULTILINE,
)

_PATTERN_SIMPLE = re.compile(
    r"^(?P<name>我|他|她|对方|.{1,8}?)[：:]\s*(?P<content>.+)$",
    re.MULTILINE,
)

_PATTERN_BRACKET = re.compile(
    r"^\[(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<name>.{1,20}?)[：:]\s*(?P<content>.+)$",
    re.MULTILINE,
)


def parse_chat_record(text: str) -> ParsedChat:
    """Try to parse text as a chat record. Returns ParsedChat with is_chat_record flag."""
    text = text.strip()

    # Try format 1: Full datetime format (WeChat copy-paste)
    # "张三 2024-01-15 10:30\n你好"
    lines = _parse_full_format(text)
    if lines:
        return ParsedChat(lines=lines, raw_text=text, is_chat_record=True)

    # Try format 2: Bracket time format
    # "[10:30] 张三：你好"
    lines = _parse_bracket_format(text)
    if lines:
        return ParsedChat(lines=lines, raw_text=text, is_chat_record=True)

    # Try format 3: Simple colon format
    # "我：你好" / "她：你好呀"
    lines = _parse_simple_format(text)
    if lines:
        return ParsedChat(lines=lines, raw_text=text, is_chat_record=True)

    # Not a recognized chat format — treat as free-form text
    return ParsedChat(lines=[], raw_text=text, is_chat_record=False)


def _parse_full_format(text: str) -> list[ChatLine]:
    """Parse 'name datetime\\ncontent' format."""
    matches = list(_PATTERN_FULL.finditer(text))
    if len(matches) < 2:
        return []

    lines = []
    for i, m in enumerate(matches):
        name = m.group("name").strip()
        time_str = m.group("time").strip()
        # Content is between end of this header and start of next header
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            lines.append(ChatLine(sender=name, time=time_str, content=content))
    return lines


def _parse_bracket_format(text: str) -> list[ChatLine]:
    """Parse '[time] name：content' format."""
    matches = list(_PATTERN_BRACKET.finditer(text))
    if len(matches) < 2:
        return []
    return [
        ChatLine(
            sender=m.group("name").strip(),
            time=m.group("time").strip(),
            content=m.group("content").strip(),
        )
        for m in matches
    ]


def _parse_simple_format(text: str) -> list[ChatLine]:
    """Parse 'name：content' format."""
    matches = list(_PATTERN_SIMPLE.finditer(text))
    if len(matches) < 2:
        return []
    return [
        ChatLine(
            sender=m.group("name").strip(),
            time="",
            content=m.group("content").strip(),
        )
        for m in matches
    ]


def format_for_ai(parsed: ParsedChat) -> str:
    """Format parsed chat into a clean representation for AI analysis."""
    if not parsed.is_chat_record or not parsed.lines:
        return parsed.raw_text

    parts = []
    for line in parsed.lines:
        time_part = f" ({line.time})" if line.time else ""
        parts.append(f"【{line.sender}】{time_part}: {line.content}")
    return "\n".join(parts)
