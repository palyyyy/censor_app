from __future__ import annotations

import threading
from typing import Optional

import customtkinter as ctk

from app.audio.device_manager import (find_device_by_name, list_input_devices,
                                      list_output_devices)
from app.audio.live_processor import LiveConfig, LiveProcessor
from app.censor import CensorList
from app.stt import get_engine
from app.stt.base import Word
from app.utils.logger import get_logger
from config import AppSettings, TARGET_SAMPLE_RATE

from .components import WordListEditor

log = get_logger(__name__)

_NO_PRESET = "—"


class LiveModeWindow(ctk.CTkToplevel):
    def __init__(self, master, settings: AppSettings) -> None:
        super().__init__(master)
        self.settings = settings
        self.censor_list = CensorList()

        self.title("Live censoring")
        self.geometry("980x720")
        self.minsize(820, 620)

        self._processor: Optional[LiveProcessor] = None
        self._start_thread: Optional[threading.Thread] = None
        self._in_display_to_name: dict[str, str] = {}
        self._out_display_to_name: dict[str, str] = {}

        self._build()
        self._refresh_devices()
        self._refresh_presets()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(16, 8))
        top.grid_columnconfigure(1, weight=1)
        top.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(top, text="Input:").grid(row=0, column=0, sticky="w")
        self._in_var = ctk.StringVar()
        self._in_menu = ctk.CTkOptionMenu(top, variable=self._in_var, values=["-"])
        self._in_menu.grid(row=0, column=1, sticky="ew", padx=(6, 12))

        ctk.CTkLabel(top, text="Output:").grid(row=0, column=2, sticky="w")
        self._out_var = ctk.StringVar()
        self._out_menu = ctk.CTkOptionMenu(top, variable=self._out_var, values=["-"])
        self._out_menu.grid(row=0, column=3, sticky="ew", padx=(6, 12))

        ctk.CTkButton(top, text="↻ Refresh", width=90, command=self._refresh_devices) \
            .grid(row=0, column=4, sticky="e")

        ctk.CTkLabel(top, text="Preset:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self._preset_var = ctk.StringVar(value=_NO_PRESET)
        self._preset_menu = ctk.CTkOptionMenu(top, variable=self._preset_var,
                                              values=[_NO_PRESET],
                                              command=self._on_preset_selected)
        self._preset_menu.grid(row=1, column=1, sticky="ew", padx=(6, 12), pady=(8, 0))

        preset_btns = ctk.CTkFrame(top, fg_color="transparent")
        preset_btns.grid(row=1, column=2, columnspan=3, sticky="w", pady=(8, 0))
        ctk.CTkButton(preset_btns, text="Save preset", width=100,
                      command=self._on_save_preset).pack(side="left", padx=(0, 6))
        ctk.CTkButton(preset_btns, text="Delete", width=70,
                      fg_color="gray30", hover_color="#a33",
                      command=self._on_delete_preset).pack(side="left")

        self._editor = WordListEditor(self, self.censor_list,
                                      on_change=self._on_list_change)
        self._editor.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 8))

        right = ctk.CTkFrame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=(0, 8))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(right, text="Controls", font=ctk.CTkFont(size=14, weight="bold")) \
            .grid(row=0, column=0, sticky="w", padx=12, pady=(12, 0))

        ctk.CTkLabel(right, text="Lookahead (s)").grid(row=1, column=0, sticky="w",
                                                       padx=12, pady=(8, 0))
        self._lookahead_var = ctk.DoubleVar(value=self.settings.lookahead_seconds)
        slider_row = ctk.CTkFrame(right, fg_color="transparent")
        slider_row.grid(row=2, column=0, sticky="ew", padx=12, pady=(2, 0))
        slider_row.grid_columnconfigure(0, weight=1)
        self._lookahead_slider = ctk.CTkSlider(
            slider_row, from_=0.5, to=5.0, variable=self._lookahead_var,
            command=self._update_lookahead_label,
        )
        self._lookahead_slider.grid(row=0, column=0, sticky="ew")
        self._lookahead_label = ctk.CTkLabel(slider_row, text="")
        self._lookahead_label.grid(row=0, column=1, padx=(6, 0))
        self._update_lookahead_label(self._lookahead_var.get())

        self._start_btn = ctk.CTkButton(right, text="Start live censoring",
                                        height=40, command=self._on_start)
        self._start_btn.grid(row=3, column=0, sticky="ew", padx=12, pady=(12, 4))
        self._stop_btn = ctk.CTkButton(right, text="Stop", height=36,
                                       fg_color="gray30", hover_color="#a33",
                                       state="disabled", command=self._on_stop)
        self._stop_btn.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 4))

        self._log_box = ctk.CTkTextbox(right, height=260)
        self._log_box.grid(row=5, column=0, sticky="nsew", padx=12, pady=(8, 12))
        self._log_box.insert("1.0", "Live transcription will appear here.\n"
                                    "Censored words are highlighted in the log.\n")
        self._log_box.configure(state="disabled")

        footer = ctk.CTkLabel(self, text=f"Engine: {self.settings.stt_engine}  ·  "
                                         f"Model: {self.settings.stt_model}",
                              text_color="gray")
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 12))

    def _update_lookahead_label(self, value: float) -> None:
        self._lookahead_label.configure(text=f"{float(value):.2f} s")

    def _refresh_devices(self) -> None:
        ins = list_input_devices()
        outs = list_output_devices()
        self._in_display_to_name = {str(d): d.name for d in ins}
        self._out_display_to_name = {str(d): d.name for d in outs}
        in_names = list(self._in_display_to_name) or ["-"]
        out_names = list(self._out_display_to_name) or ["-"]
        self._in_menu.configure(values=in_names)
        self._out_menu.configure(values=out_names)
        # Default selection = system default (first entry)
        if not self._in_var.get() or self._in_var.get() not in in_names:
            self._in_var.set(in_names[0])
        if not self._out_var.get() or self._out_var.get() not in out_names:
            self._out_var.set(out_names[0])

    def _parse_device_index(self, display: str) -> Optional[int]:
        try:
            closing = display.index("]")
            return int(display[1:closing])
        except Exception:
            return None


    def _refresh_presets(self) -> None:
        names = sorted(self.settings.live_presets)
        self._preset_menu.configure(values=names or [_NO_PRESET])
        if self._preset_var.get() not in names:
            self._preset_var.set(names[0] if names else _NO_PRESET)

    def _on_preset_selected(self, name: str) -> None:
        preset = self.settings.live_presets.get(name)
        if not preset:
            return
        self._refresh_devices()
        self._select_device(self._in_var, self._in_menu, preset.get("input"), "input")
        self._select_device(self._out_var, self._out_menu, preset.get("output"), "output")
        if "lookahead" in preset:
            self._lookahead_var.set(float(preset["lookahead"]))
            self._update_lookahead_label(preset["lookahead"])
        self._append_log(f"Preset applied: {name}")

    def _select_device(self, var: ctk.StringVar, menu: ctk.CTkOptionMenu,
                       name: str | None, kind: str) -> None:
        if not name:
            return
        device = find_device_by_name(name, kind)
        if device is None:
            self._append_log(f"Preset {kind} device not found: {name}")
            return
        var.set(str(device))

    def _on_save_preset(self) -> None:
        dialog = ctk.CTkInputDialog(text="Preset name:", title="Save preset")
        name = (dialog.get_input() or "").strip()
        if not name:
            return
        self.settings.live_presets[name] = {
            "input": self._in_display_to_name.get(self._in_var.get()),
            "output": self._out_display_to_name.get(self._out_var.get()),
            "lookahead": float(self._lookahead_var.get()),
        }
        self.settings.save()
        self._refresh_presets()
        self._preset_var.set(name)
        self._append_log(f"Preset saved: {name}")

    def _on_delete_preset(self) -> None:
        name = self._preset_var.get()
        if name not in self.settings.live_presets:
            return
        del self.settings.live_presets[name]
        self.settings.save()
        self._refresh_presets()
        self._append_log(f"Preset deleted: {name}")


    def _on_list_change(self) -> None:
        if self._processor:
            self._processor.matcher.refresh()

    def _append_log(self, msg: str) -> None:
        self._log_box.configure(state="normal")
        self._log_box.insert("end", msg + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _on_word(self, word: Word, censored: bool) -> None:
        tag = "[CENSORED] " if censored else "           "
        self.after(0, self._append_log, f"{tag}{word.start:6.2f}s  {word.text}")

    def _on_start(self) -> None:
        if self._processor is not None:
            return

        in_dev = self._parse_device_index(self._in_var.get())
        out_dev = self._parse_device_index(self._out_var.get())

        cfg = LiveConfig(
            sample_rate=TARGET_SAMPLE_RATE,
            chunk_seconds=self.settings.chunk_seconds,
            lookahead_seconds=float(self._lookahead_var.get()),
            input_device=in_dev,
            output_device=out_dev,
            sfx_tail=self.settings.sfx_tail,
        )

        self._start_btn.configure(state="disabled", text="Loading model...")
        self._append_log("--- Starting live mode ---")

        self._start_thread = threading.Thread(
            target=self._start_worker, args=(cfg,), daemon=True,
        )
        self._start_thread.start()

    def _start_worker(self, cfg: LiveConfig) -> None:
        try:
            engine = get_engine(
                self.settings.stt_engine,
                self.settings.stt_model,
                self.settings.language,
            )
            proc = LiveProcessor(engine, self.censor_list, cfg)
            proc.on_word = self._on_word
            proc.start()
            self._processor = proc
            self.after(0, self._start_success)
        except Exception as e:
            log.exception("Failed to start live mode")
            self.after(0, self._start_failed, str(e))

    def _start_success(self) -> None:
        self._start_btn.configure(state="disabled", text="Running...")
        self._stop_btn.configure(state="normal")
        self._append_log("Live censoring active.")

    def _start_failed(self, msg: str) -> None:
        self._append_log(f"ERROR: {msg}")
        self._start_btn.configure(state="normal", text="Start live censoring")

    def _on_stop(self) -> None:
        if self._processor is None:
            return
        self._processor.stop()
        self._processor = None
        self._stop_btn.configure(state="disabled")
        self._start_btn.configure(state="normal", text="Start live censoring")
        self._append_log("Stopped.")

    def destroy(self) -> None:
        if self._processor is not None:
            try:
                self._processor.stop()
            except Exception:
                pass
        super().destroy()
