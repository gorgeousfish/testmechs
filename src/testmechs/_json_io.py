"""Atomic strict-JSON file writer for testmechs report persistence.

This module provides a single utility function that serializes a Python
object to a JSON file using an atomic write pattern (write-to-temp then
rename). Non-finite floats (NaN, Infinity) are rejected at serialization
time so that all persisted artifacts remain valid strict JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any


def write_strict_json_atomic(path: Path, payload: Any) -> None:
    """Serialize *payload* to *path* as strict JSON using an atomic write.

    The function writes to a temporary file in the same directory as *path*
    and then atomically renames it to *path*, ensuring readers never observe
    a partially-written file.

    Parameters
    ----------
    path : Path
        Destination file path.  Parent directories are created if absent.
    payload : Any
        JSON-serializable object.  Must not contain ``NaN`` or ``Infinity``
        values (``allow_nan=False`` is enforced).

    Raises
    ------
    ValueError
        If *payload* contains non-finite numeric values.
    OSError
        If the temporary file cannot be written or renamed.

    Examples
    --------
    >>> from pathlib import Path
    >>> write_strict_json_atomic(Path("/tmp/result.json"), {"score": 0.95})
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
