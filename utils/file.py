import os
from typing import Iterable

def check_ext(filename: str, allowed_extensions: Iterable[str]) -> bool:
    """
    Check if the file has an allowed extension.

    Args:
        filename (str): Name or path of the file.
        allowed_extensions (Iterable[str]): Collection of allowed extensions 
                                            (e.g., {"jpg", "png", "wav"}).

    Returns:
        bool: True if the file extension is in allowed_extensions, False otherwise.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in allowed_extensions


def get_name_without_ext(filename: str) -> str:
    """
    Get the filename without its extension.

    Args:
        filename (str): Name or path of the file.

    Returns:
        str: Filename without the extension.

    Example:
        >>> get_name_without_ext("example.docx")
        'example'
    """
    return os.path.splitext(os.path.basename(filename))[0]
