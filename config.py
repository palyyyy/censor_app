from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

APP_NAME = "CensorApp"
APP_VERSION = "0.1.0"


USER_CONFIG_DIR = Path.home() / ".censor_app"
USER_CONFIG_FILE = USER_CONFIG_DIR / "settings.json"
DEFAULT_SFX_DIR = USER_CONFIG_DIR / "sfx"
TARGET_SAMPLE_RATE = 16_000          
OUTPUT_SAMPLE_RATE = 44_100          
BEEP_FREQ_HZ = 1000.0
BEEP_AMPLITUDE = 0.35                


@dataclass
class AppSettings:
    """User-configurable settings. Persisted to JSON."""

    stt_engine: str = "faster-whisper"        
    stt_model: str = "small.en"               
    language: str = "en"                      

    lookahead_seconds: float = 2.0            
    chunk_seconds: float = 1.0                
    input_device: str | None = None           
    output_device: str | None = None

    default_mode: str = "beep"                
    default_sfx_path: str | None = None       

    appearance: str = "system"                
    color_theme: str = "blue"

    def save(self) -> None:
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        USER_CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "AppSettings":
        if not USER_CONFIG_FILE.exists():
            return cls()
        try:
            data: dict[str, Any] = json.loads(USER_CONFIG_FILE.read_text())
            known = {f.name for f in cls.__dataclass_fields__.values()}
            return cls(**{k: v for k, v in data.items() if k in known})
        except Exception:
            return cls()
