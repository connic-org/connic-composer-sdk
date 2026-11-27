from pathlib import Path

PROJECT_DIRECTORIES = ("agents", "tools", "middleware", "schemas", "guardrails", "hooks", "tests", "connectors", "assets")
# Audio files a project may contain, such as a voice connector's hold audio.
AUDIO_EXTENSIONS = (".mp3", ".wav")
# Documents a deploy adds to the environment's Retrieval; folders become namespaces.
RETRIEVAL_DIRECTORY = "retrieval"
RETRIEVAL_EXTENSIONS = (
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".txt", ".md", ".markdown", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".log",
    ".docx", ".doc", ".odt", ".rtf", ".html", ".htm",
    ".xlsx", ".xls", ".ods", ".pptx", ".odp", ".eml", ".msg",
)
MAX_RETRIEVAL_DEPTH = 10


def retrieval_path_error(path: Path) -> str | None:
    """Why the file at ``path`` under retrieval/ can't become a retrieval entry, or None."""
    folders = path.parts[:-1]
    if path.suffix.lower() not in RETRIEVAL_EXTENSIONS:
        return f"File type '{path.suffix or path.name}' can't be added to retrieval. Allowed: {', '.join(RETRIEVAL_EXTENSIONS)}"
    if len(folders) > MAX_RETRIEVAL_DEPTH:
        return f"Folders are nested deeper than {MAX_RETRIEVAL_DEPTH} levels"
    if any("." in folder for folder in folders):
        return "Folder names can't contain dots, they become namespace levels"
    return None


def contained_source(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Source path '{path}' is outside '{root}'")
    return resolved


def project_destination(path: Path, root: Path) -> Path:
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError:
        raise ValueError(f"Destination path '{path}' is outside '{root}'") from None
    if ".." in relative.parts:
        raise ValueError(f"Destination path '{path}' is outside '{root}'")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Symbolic links are not allowed: {current}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Destination path '{path}' is outside '{root}'")
    return path
