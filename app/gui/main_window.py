from __future__ import annotations

import customtkinter as ctk

from config import APP_NAME, APP_VERSION, AppSettings

from .file_mode_window import FileModeWindow
from .live_mode_window import LiveModeWindow
from .settings_window import SettingsWindow


class MainWindow(ctk.CTk):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings

        ctk.set_appearance_mode(settings.appearance)
        ctk.set_default_color_theme(settings.color_theme)

        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("760x520")
        self.minsize(640, 440)

        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        # Top bar
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 0))
        top.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            top, text=APP_NAME,
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        gear = ctk.CTkButton(top, text="⚙  Settings", width=110, command=self._open_settings)
        gear.grid(row=0, column=1, sticky="e")

        subtitle = ctk.CTkLabel(
            top, text="Speech-to-text powered word censoring",
            text_color="gray", font=ctk.CTkFont(size=13),
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(2, 0))

        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        cards.grid_columnconfigure(0, weight=1, uniform="cards")
        cards.grid_columnconfigure(1, weight=1, uniform="cards")
        cards.grid_rowconfigure(0, weight=1)

        file_card = self._card(
            cards,
            title="Pre-recorded file",
            subtitle="Upload an mp3 / wav file and export a censored\nversion + transcript.",
            button_text="Open file mode",
            command=self._open_file_mode,
        )
        file_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        live_card = self._card(
            cards,
            title="Live censoring",
            subtitle="Route a microphone to a speaker or virtual\ndevice with live beeping of banned words.",
            button_text="Open live mode",
            command=self._open_live_mode,
        )
        live_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        footer = ctk.CTkLabel(
            self,
            text=f"STT engine: {self.settings.stt_engine}  ·  Model: {self.settings.stt_model}",
            text_color="gray",
        )
        footer.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 12))
        self._footer = footer

    def _card(self, master, title: str, subtitle: str,
              button_text: str, command) -> ctk.CTkFrame:
        card = ctk.CTkFrame(master, corner_radius=12)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=0)
        card.grid_rowconfigure(2, weight=0)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.grid(row=0, column=0, sticky="nsew", padx=20, pady=(28, 8))
        inner.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            inner, text=title,
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            inner, text=subtitle, text_color="gray", justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        ctk.CTkButton(
            card, text=button_text, height=40, command=command,
        ).grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))
        return card

    def _refresh_footer(self) -> None:
        self._footer.configure(
            text=f"STT engine: {self.settings.stt_engine}  ·  Model: {self.settings.stt_model}",
        )

    def _open_settings(self) -> None:
        win = SettingsWindow(self, self.settings, on_save=self._refresh_footer)
        win.grab_set()

    def _open_file_mode(self) -> None:
        win = FileModeWindow(self, self.settings)
        win.grab_set()

    def _open_live_mode(self) -> None:
        win = LiveModeWindow(self, self.settings)
        win.grab_set()
