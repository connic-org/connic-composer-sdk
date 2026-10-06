import re


def escape_terminal_text(text: str) -> str:
    """Render C0, DEL and C1 controls as visible text before terminal styling."""
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", lambda match: f"\\x{ord(match.group()):02x}", text)
