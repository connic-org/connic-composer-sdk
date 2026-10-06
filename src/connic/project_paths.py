from pathlib import Path


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
