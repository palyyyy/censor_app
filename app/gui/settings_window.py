from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from app.stt import list_engines
from config import AppSettings


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, master, settings: AppSettings,
                 on_save: Callable[[], None] | None = None) -> None:
        super().__init__(master)
        self.settings = settings
        self.on_save = on_save

        self.title("Settings")
        self.geometry("520x600")
        self.resizable(False, False)

        self.grid_columnconfigure(0, weight=1)

        self._build()
        self._populate()

    def _build(self) -> None:
        pad = dict(padx=20, pady=(10, 0))

        ctk.CTkLabel(self, text="STT Engine",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w", **pad)

        self._engines = [e for e in list_engines(only_available=False)]
        self._engine_choices = [e["display_name"] + (" — unavailable" if not e["available"] else "")
                                for e in self._engines]
        self._engine_var = ctk.StringVar()
        self._engine_menu = ctk.CTkOptionMenu(
            self, variable=self._engine_var, values=self._engine_choices,
            command=self._on_engine_change,
        )
        self._engine_menu.grid(row=1, column=0, sticky="ew", padx=20, pady=(4, 0))

        ctk.CTkLabel(self, text="Model",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(row=2, column=0, sticky="w", **pad)
        self._model_var = ctk.StringVar()
        self._model_menu = ctk.CTkOptionMenu(self, variable=self._model_var, values=["-"])
        self._model_menu.grid(row=3, column=0, sticky="ew", padx=20, pady=(4, 0))

        ctk.CTkLabel(self, text="Language",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(row=4, column=0, sticky="w", **pad)
        self._lang_var = ctk.StringVar(value=self.settings.language)
        ctk.CTkEntry(self, textvariable=self._lang_var,
                     placeholder_text="en, de, fr, auto, ...") \
            .grid(row=5, column=0, sticky="ew", padx=20, pady=(4, 0))

        ctk.CTkLabel(self, text="Live lookahead (seconds)",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(row=6, column=0, sticky="w", **pad)
        self._lookahead_var = ctk.DoubleVar(value=self.settings.lookahead_seconds)
        self._lookahead_slider = ctk.CTkSlider(
            self, from_=0.5, to=5.0, variable=self._lookahead_var,
            command=self._update_lookahead_label,
        )
        self._lookahead_slider.grid(row=7, column=0, sticky="ew", padx=20, pady=(4, 0))
        self._lookahead_label = ctk.CTkLabel(self, text="")
        self._lookahead_label.grid(row=8, column=0, sticky="w", padx=20)
        self._update_lookahead_label(self._lookahead_var.get())

        ctk.CTkLabel(self, text="Sound effects",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(row=9, column=0, sticky="w", **pad)
        self._sfx_tail_var = ctk.BooleanVar(value=self.settings.sfx_tail)
        ctk.CTkSwitch(self, text="Let an SFX play past the censored word",
                      variable=self._sfx_tail_var) \
            .grid(row=10, column=0, sticky="w", padx=20, pady=(4, 0))

        ctk.CTkLabel(self, text="Appearance",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(row=11, column=0, sticky="w", **pad)
        self._appearance_var = ctk.StringVar(value=self.settings.appearance)
        ctk.CTkOptionMenu(self, variable=self._appearance_var,
                          values=["system", "light", "dark"],
                          command=lambda v: ctk.set_appearance_mode(v)) \
            .grid(row=12, column=0, sticky="ew", padx=20, pady=(4, 0))

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.grid(row=13, column=0, sticky="ew", padx=20, pady=16)
        btns.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(btns, text="Cancel", fg_color="gray30", hover_color="gray25",
                      command=self.destroy).grid(row=0, column=0, sticky="e", padx=(0, 8))
        ctk.CTkButton(btns, text="Save", command=self._save) \
            .grid(row=0, column=1, sticky="e")

    def _update_lookahead_label(self, value: float) -> None:
        self._lookahead_label.configure(text=f"{float(value):.2f} s")

    def _populate(self) -> None:
        current_name = self.settings.stt_engine
        current_display = None
        for e, label in zip(self._engines, self._engine_choices):
            if e["name"] == current_name:
                current_display = label
                break
        if current_display is None and self._engine_choices:
            current_display = self._engine_choices[0]
        if current_display:
            self._engine_var.set(current_display)
            self._on_engine_change(current_display)

    def _selected_engine_entry(self) -> dict:
        choice = self._engine_var.get()
        idx = self._engine_choices.index(choice)
        return self._engines[idx]

    def _on_engine_change(self, _choice: str) -> None:
        entry = self._selected_engine_entry()
        models = entry["models"] or ["-"]
        self._model_menu.configure(values=models)
        if self.settings.stt_model in models:
            self._model_var.set(self.settings.stt_model)
        else:
            self._model_var.set(models[0])

    def _save(self) -> None:
        entry = self._selected_engine_entry()
        self.settings.stt_engine = entry["name"]
        self.settings.stt_model = self._model_var.get()
        self.settings.language = self._lang_var.get().strip() or "en"
        self.settings.lookahead_seconds = float(self._lookahead_var.get())
        self.settings.sfx_tail = bool(self._sfx_tail_var.get())
        self.settings.appearance = self._appearance_var.get()
        self.settings.save()
        if self.on_save:
            self.on_save()
        self.destroy()
