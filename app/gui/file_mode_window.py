from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from app.audio.file_processor import process_file
from app.censor import CensorList
from app.stt import get_engine
from app.utils.logger import get_logger
from config import AppSettings

from .components import WordListEditor

log = get_logger(__name__)


class FileModeWindow(ctk.CTkToplevel):
    def __init__(self, master, settings: AppSettings) -> None:
        super().__init__(master)
        self.settings = settings
        self.censor_list = CensorList()

        self.title("Pre-recorded file mode")
        self.geometry("900x640")
        self.minsize(760, 560)

        self._input_path: Path | None = None
        self._worker: threading.Thread | None = None

        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        # Top bar: input file + output format
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(16, 8))
        top.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(top, text="Pick audio file", width=140, command=self._on_pick_file) \
            .grid(row=0, column=0, sticky="w")
        self._path_label = ctk.CTkLabel(top, text="No file selected", text_color="gray",
                                        anchor="w")
        self._path_label.grid(row=0, column=1, sticky="ew", padx=(12, 12))

        self._format_var = ctk.StringVar(value="wav")
        ctk.CTkLabel(top, text="Output format:").grid(row=0, column=2, sticky="e")
        ctk.CTkOptionMenu(top, variable=self._format_var, values=["wav", "mp3"], width=80) \
            .grid(row=0, column=3, padx=(6, 0))

        # Left panel: word list editor
        self._editor = WordListEditor(self, self.censor_list)
        self._editor.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 8))

        right = ctk.CTkFrame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=(0, 8))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(right, text="Run", font=ctk.CTkFont(size=14, weight="bold")) \
            .grid(row=0, column=0, sticky="w", padx=12, pady=(12, 0))

        self._run_btn = ctk.CTkButton(right, text="Transcribe + Censor",
                                      height=40, command=self._on_run)
        self._run_btn.grid(row=1, column=0, sticky="ew", padx=12, pady=(8, 4))

        self._progress = ctk.CTkProgressBar(right, mode="indeterminate")
        self._progress.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 4))
        self._progress.stop()
        self._progress.set(0)

        self._status_box = ctk.CTkTextbox(right, height=260)
        self._status_box.grid(row=3, column=0, sticky="nsew", padx=12, pady=(4, 12))
        self._status_box.insert("1.0", "Status will appear here.\n")
        self._status_box.configure(state="disabled")

        footer = ctk.CTkLabel(self, text=f"Engine: {self.settings.stt_engine}  ·  "
                                         f"Model: {self.settings.stt_model}",
                              text_color="gray")
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 12))

    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self._status_box.configure(state="normal")
        self._status_box.insert("end", msg + "\n")
        self._status_box.see("end")
        self._status_box.configure(state="disabled")

    def _on_pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Pick audio file",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*")],
        )
        if not path:
            return
        self._input_path = Path(path)
        self._path_label.configure(text=self._input_path.name, text_color=None)

    def _on_run(self) -> None:
        if self._input_path is None:
            messagebox.showerror("No input", "Please pick an audio file first.")
            return
        if self._worker is not None and self._worker.is_alive():
            return

        out_ext = "." + self._format_var.get()
        out_audio = filedialog.asksaveasfilename(
            title="Save censored audio as",
            defaultextension=out_ext,
            initialfile=self._input_path.stem + "_censored" + out_ext,
            filetypes=[("WAV", "*.wav"), ("MP3", "*.mp3")],
        )
        if not out_audio:
            return
        out_audio = Path(out_audio)
        out_txt = out_audio.with_suffix(".txt")

        self._run_btn.configure(state="disabled", text="Working...")
        self._progress.start()
        self._log(f"--- Starting: {self._input_path.name} ---")

        self._worker = threading.Thread(
            target=self._run_worker,
            args=(self._input_path, out_audio, out_txt),
            daemon=True,
        )
        self._worker.start()

    def _run_worker(self, inp: Path, out_audio: Path, out_txt: Path) -> None:
        try:
            engine = get_engine(
                self.settings.stt_engine,
                self.settings.stt_model,
                self.settings.language,
            )
            result = process_file(
                inp, out_audio, out_txt,
                engine=engine,
                censor_list=self.censor_list,
                progress_cb=lambda m: self.after(0, self._log, m),
            )
            self.after(0, self._on_done, result.audio_out_path, result.transcript_path,
                       len(result.censored_words))
        except Exception as e:
            log.exception("File processing failed")
            self.after(0, self._on_error, str(e))

    def _on_done(self, audio_path: Path, txt_path: Path, n_censored: int) -> None:
        self._progress.stop()
        self._progress.set(1.0)
        self._run_btn.configure(state="normal", text="Transcribe + Censor")
        self._log(f"Censored {n_censored} word(s).")
        self._log(f"Saved audio:      {audio_path}")
        self._log(f"Saved transcript: {txt_path}")
        messagebox.showinfo("Done", f"Saved censored audio and transcript.\n\n"
                                     f"Audio:\n{audio_path}\n\nTranscript:\n{txt_path}")

    def _on_error(self, msg: str) -> None:
        self._progress.stop()
        self._run_btn.configure(state="normal", text="Transcribe + Censor")
        self._log(f"ERROR: {msg}")
        messagebox.showerror("Processing failed", msg)
