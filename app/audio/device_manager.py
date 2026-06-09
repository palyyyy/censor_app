from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sounddevice as sd


@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0

    def __str__(self) -> str:
        return f"[{self.index}] {self.name}"


def list_devices() -> list[AudioDevice]:
    devices: list[AudioDevice] = []
    for i, d in enumerate(sd.query_devices()):
        devices.append(AudioDevice(
            index=i,
            name=d.get("name", f"device {i}"),
            max_input_channels=int(d.get("max_input_channels", 0)),
            max_output_channels=int(d.get("max_output_channels", 0)),
            default_samplerate=float(d.get("default_samplerate", 44100.0)),
        ))
    return devices


def list_input_devices() -> list[AudioDevice]:
    return [d for d in list_devices() if d.is_input]


def list_output_devices() -> list[AudioDevice]:
    return [d for d in list_devices() if d.is_output]


def find_device_by_name(name: str, kind: str = "input") -> Optional[AudioDevice]:
    for d in list_devices():
        if d.name == name and (d.is_input if kind == "input" else d.is_output):
            return d
    return None
