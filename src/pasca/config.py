"""Endpoint settings, stored in ~/.config/pasca/config.toml."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

API_KEY_ENV = "PASCA_API_KEY"


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "pasca" / "config.toml"


@dataclass
class Settings:
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key: str = ""
    save_api_key: bool = False
    temperature: float = 0.3
    max_excerpt_chars: int = 60_000

    def effective_api_key(self) -> str:
        return self.api_key or os.environ.get(API_KEY_ENV, "")


def load(path: Path | None = None) -> Settings:
    path = path or config_path()
    settings = Settings()
    if not path.is_file():
        return settings
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return settings
    for key, default in asdict(settings).items():
        if key in data and isinstance(data[key], type(default)):
            setattr(settings, key, data[key])
        elif key in data and isinstance(default, float) and isinstance(data[key], int):
            setattr(settings, key, float(data[key]))
    return settings


def save(settings: Settings, path: Path | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(settings)
    if not settings.save_api_key:
        data["api_key"] = ""
    lines = []
    for key, value in data.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f"{key} = {json.dumps(value)}")  # JSON strings are valid TOML basic strings
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if settings.save_api_key:
        path.chmod(0o600)
    return path
