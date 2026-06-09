from __future__ import annotations

from pathlib import Path
from typing import Callable

import customtkinter as ctk
from tkinter import filedialog

from app.censor.censor_rules import CensorList, CensorMode, CensorRule


_MODE_LABELS = {
    CensorMode.BEEP: "Beep",
    CensorMode.SILENCE: "Silence",
    CensorMode.SFX: "SFX",
}
_MODE_FROM_LABEL = {v: k for k, v in _MODE_LABELS.items()}


class WordListEditor(ctk.CTkFrame):
    def __init__(self, master, censor_list: CensorList,
                 on_change: Callable[[], None] | None = None, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._censor_list = censor_list
        self._on_change = on_change

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkLabel(self, text="Censored words",
                              font=ctk.CTkFont(size=14, weight="bold"))
        header.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))

        # Scrollable rows container
        self._rows_frame = ctk.CTkScrollableFrame(self, label_text="")
        self._rows_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self._rows_frame.grid_columnconfigure(0, weight=1)

        # Add-row controls
        add_frame = ctk.CTkFrame(self, fg_color="transparent")
        add_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        add_frame.grid_columnconfigure(0, weight=1)

        self._new_word_var = ctk.StringVar()
        self._new_word_entry = ctk.CTkEntry(add_frame, textvariable=self._new_word_var,
                                            placeholder_text="Add a word and press Enter")
        self._new_word_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._new_word_entry.bind("<Return>", self._on_add)

        self._new_mode_var = ctk.StringVar(value=_MODE_LABELS[CensorMode.BEEP])
        self._new_mode_menu = ctk.CTkOptionMenu(
            add_frame, variable=self._new_mode_var,
            values=list(_MODE_LABELS.values()), width=100,
        )
        self._new_mode_menu.grid(row=0, column=1, padx=(0, 6))

        ctk.CTkButton(add_frame, text="+ Add", width=70, command=self._on_add) \
            .grid(row=0, column=2)

        bulk_frame = ctk.CTkFrame(self, fg_color="transparent")
        bulk_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        ctk.CTkButton(bulk_frame, text="Import from TXT",
                      width=120, command=self._on_import).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bulk_frame, text="Clear all",
                      width=90, fg_color="gray30", hover_color="gray25",
                      command=self._on_clear).pack(side="left")

        self._render_rows()

    def _render_rows(self) -> None:
        for widget in self._rows_frame.winfo_children():
            widget.destroy()

        if not self._censor_list.rules:
            empty = ctk.CTkLabel(self._rows_frame, text="No censored words yet.",
                                 text_color="gray")
            empty.grid(row=0, column=0, padx=8, pady=8, sticky="w")
            return

        for i, rule in enumerate(self._censor_list.rules):
            self._render_row(i, rule)

    def _render_row(self, index: int, rule: CensorRule) -> None:
        row = ctk.CTkFrame(self._rows_frame, fg_color="transparent")
        row.grid(row=index, column=0, sticky="ew", padx=2, pady=2)
        row.grid_columnconfigure(0, weight=1)

        word_lbl = ctk.CTkLabel(row, text=rule.word, anchor="w")
        word_lbl.grid(row=0, column=0, sticky="ew", padx=(4, 6))

        mode_var = ctk.StringVar(value=_MODE_LABELS[rule.mode])
        mode_menu = ctk.CTkOptionMenu(
            row, variable=mode_var, values=list(_MODE_LABELS.values()),
            width=90,
            command=lambda val, r=rule: self._on_mode_change(r, val),
        )
        mode_menu.grid(row=0, column=1, padx=(0, 6))

        sfx_btn_text = Path(rule.sfx_path).name if rule.sfx_path else "Pick SFX"
        sfx_btn = ctk.CTkButton(
            row, text=sfx_btn_text, width=110,
            command=lambda r=rule: self._on_pick_sfx(r),
        )
        sfx_btn.grid(row=0, column=2, padx=(0, 6))
        # Dim the button when not in SFX mode
        if rule.mode != CensorMode.SFX:
            sfx_btn.configure(state="disabled")

        remove_btn = ctk.CTkButton(
            row, text="✕", width=36, fg_color="gray30", hover_color="#a33",
            command=lambda r=rule: self._on_remove(r),
        )
        remove_btn.grid(row=0, column=3)

    def _changed(self) -> None:
        if self._on_change:
            self._on_change()

    def _on_add(self, *_args) -> None:
        word = self._new_word_var.get().strip()
        if not word:
            return
        mode = _MODE_FROM_LABEL[self._new_mode_var.get()]
        self._censor_list.add(CensorRule(word=word, mode=mode))
        self._new_word_var.set("")
        self._render_rows()
        self._changed()

    def _on_remove(self, rule: CensorRule) -> None:
        self._censor_list.remove(rule.word)
        self._render_rows()
        self._changed()

    def _on_mode_change(self, rule: CensorRule, label: str) -> None:
        new_mode = _MODE_FROM_LABEL[label]
        rule.mode = new_mode
        if new_mode == CensorMode.SFX and not rule.sfx_path:
            # Prompt immediately for an SFX file.
            path = filedialog.askopenfilename(
                title=f"Pick an SFX file for '{rule.word}'",
                filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*")],
            )
            if path:
                rule.sfx_path = path
            else:
                rule.mode = CensorMode.BEEP
        self._render_rows()
        self._changed()

    def _on_pick_sfx(self, rule: CensorRule) -> None:
        path = filedialog.askopenfilename(
            title=f"Pick an SFX file for '{rule.word}'",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*")],
        )
        if path:
            rule.sfx_path = path
            rule.mode = CensorMode.SFX
            self._render_rows()
            self._changed()

    def _on_import(self) -> None:
        path = filedialog.askopenfilename(
            title="Import words (one per line)",
            filetypes=[("Text", "*.txt"), ("All files", "*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w:
                        self._censor_list.add(CensorRule(word=w, mode=CensorMode.BEEP))
        except Exception:
            return
        self._render_rows()
        self._changed()

    def _on_clear(self) -> None:
        self._censor_list.clear()
        self._render_rows()
        self._changed()
