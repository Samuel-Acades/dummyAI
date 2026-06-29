import os
import re
from typing import Dict


def _resolve_path(path_value: str, default_path: str) -> str:
    if path_value:
        if os.path.isabs(path_value) or path_value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", path_value):
            return path_value
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path_value))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", default_path))


def get_runtime_settings() -> Dict[str, object]:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    port_value = os.getenv("PORT", os.getenv("APP_PORT", "8001"))
    port = int(port_value)

    chroma_db_path = _resolve_path(os.getenv("CHROMA_DB_PATH"), "chroma_db")
    data_dir = _resolve_path(os.getenv("DATA_DIR"), "data")

    return {
        "port": port,
        "chroma_db_path": chroma_db_path,
        "data_dir": data_dir,
    }
