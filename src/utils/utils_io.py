"""I/O helpers shared across the BrainST codebase."""

from __future__ import annotations

import json
from typing import Any


def load_json(path: str | None) -> Any | None:
    """Load a JSON file into a Python object.

    Args:
        path: Path to the JSON file. If ``None``, nothing is read.

    Returns:
        The parsed JSON content, or ``None`` if ``path`` is ``None``.

    Raises:
        FileNotFoundError: If ``path`` is given but does not exist.
        json.JSONDecodeError: If the file does not contain valid JSON.
    """
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def save_json(data: Any, path: str) -> None:
    """Save a Python object as a JSON file.

    Args:
        data: The Python object to serialize.
        path: Path where the JSON file will be saved.

    Raises:
        IOError: If the file cannot be written.
    """
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)