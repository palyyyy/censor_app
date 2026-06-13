from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from app.audio.file_processor import process_file
from app.audio.sfx_library import stop_preview
from app.censor import CensorList, find_word_occurrences, format_timestamp
from app.censor.effects import EffectOptions
from app.stt import Transcript, get_engine
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
        self.geometry("900x680")
        self.minsize(760, 600)

        self._input_path: Path | None = None
        self._worker: threading.Thread | None = None
        self._cached_transcript: Transcript | None = None
        self._cached_for: Path | None = None

        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

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

        self._editor = WordListEditor(self, self.censor_list)
        self._editor.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 8))

        right = ctk.CTkFrame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=(0, 8))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(6, weight=1)

        ctk.CTkLabel(right, text="Run", font=ctk.CTkFont(size=14, weight="bold")) \
            .grid(row=0, column=0, sticky="w", padx=12, pady=(12, 0))

        self._run_btn = ctk.CTkButton(right, text="Transcribe + Censor",
                                      height=40, command=self._on_run)
        self._run_btn.grid(row=1, column=0, sticky="ew", padx=12, pady=(8, 4))

        self._progress = ctk.CTkProgressBar(right, mode="indeterminate")
        self._progress.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 4))
        self._progress.stop()
        self._progress.set(0)

        ctk.CTkLabel(right, text="Search word", font=ctk.CTkFont(size=14, weight="bold")) \
            .grid(row=3, column=0, sticky="w", padx=12, pady=(8, 0))

        search_row = ctk.CTkFrame(right, fg_color="transparent")
        search_row.grid(row=4, column=0, sticky="ew", padx=12, pady=(4, 0))
        search_row.grid_columnconfigure(0, weight=1)
        self._search_var = ctk.StringVar()
        self._search_entry = ctk.CTkEntry(search_row, textvariable=self._search_var,
                                          placeholder_text="Word to find")
        self._search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._search_entry.bind("<Return>", self._on_search)
        self._search_btn = ctk.CTkButton(search_row, text="Find", width=70,
                                         command=self._on_search)
        self._search_btn.grid(row=0, column=1)

        self._status_box = ctk.CTkTextbox(right, height=240)
        self._status_box.grid(row=6, column=0, sticky="nsew", padx=12, pady=(8, 12))
        self._status_box.insert("1.0", "Status will appear here.\n")
        self._status_box.configure(state="disabled")

        footer = ctk.CTkLabel(self, text=f"Engine: {self.settings.stt_engine}  ·  "
                                         f"Model: {self.settings.stt_model}",
                              text_color="gray")
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 12))


    def _log(self, msg: str) -> None:
        self._status_box.configure(state="normal")
        self._status_box.insert("end", msg + "\n")
        self._status_box.see("end")
        self._status_box.configure(state="disabled")

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _set_working(self, working: bool) -> None:
        state = "disabled" if working else "normal"
        self._run_btn.configure(state=state,
                                text="Working..." if working else "Transcribe + Censor")
        self._search_btn.configure(state=state)
        if working:
            self._progress.start()
        else:
            self._progress.stop()

    def _cached_transcript_for(self, path: Path) -> Transcript | None:
        if self._cached_transcript is not None and self._cached_for == path:
            return self._cached_transcript
        return None

    def _on_pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Pick audio file",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*")],
        )
        if not path:
            return
        self._input_path = Path(path)
        self._path_label.configure(
            text=self._input_path.name,
            text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"],
        )
        self._cached_transcript = None
        self._cached_for = None


    def _on_run(self) -> None:
        if self._input_path is None:
            messagebox.showerror("No input", "Please pick an audio file first.")
            return
        if self._busy():
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

        self._set_working(True)
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
                effect_options=EffectOptions(sfx_tail=self.settings.sfx_tail),
                transcript=self._cached_transcript_for(inp),
            )
            self._cached_transcript = result.transcript
            self._cached_for = inp
            self.after(0, self._on_done, result.audio_out_path, result.transcript_path,
                       len(result.censored_words))
        except Exception as e:
            log.exception("File processing failed")
            self.after(0, self._on_error, str(e))

    def _on_done(self, audio_path: Path, txt_path: Path, n_censored: int) -> None:
        self._set_working(False)
        self._progress.set(1.0)
        self._log(f"Censored {n_censored} word(s).")
        self._log(f"Saved audio:      {audio_path}")
        self._log(f"Saved transcript: {txt_path}")
        messagebox.showinfo("Done", f"Saved censored audio and transcript.\n\n"
                                     f"Audio:\n{audio_path}\n\nTranscript:\n{txt_path}")

    def _on_error(self, msg: str) -> None:
        self._set_working(False)
        self._log(f"ERROR: {msg}")
        messagebox.showerror("Processing failed", msg)


    def _on_search(self, *_args) -> None:
        if self._input_path is None:
            messagebox.showerror("No input", "Please pick an audio file first.")
            return
        query = self._search_var.get().strip()
        if not query or self._busy():
            return

        cached = self._cached_transcript_for(self._input_path)
        if cached is not None:
            self._show_search_results(query, cached)
            return

        self._set_working(True)
        self._log(f'--- Searching for "{query}" in {self._input_path.name} ---')
        self._worker = threading.Thread(
            target=self._search_worker, args=(self._input_path, query), daemon=True,
        )
        self._worker.start()

    def _search_worker(self, path: Path, query: str) -> None:
        try:
            engine = get_engine(
                self.settings.stt_engine,
                self.settings.stt_model,
                self.settings.language,
            )
            self.after(0, self._log, "Transcribing... (this can take a while for large files)")
            transcript = engine.transcribe_file(path)
            self._cached_transcript = transcript
            self._cached_for = path
            self.after(0, self._search_done, query, transcript)
        except Exception as e:
            log.exception("Word search failed")
            self.after(0, self._on_error, str(e))

    def _search_done(self, query: str, transcript: Transcript) -> None:
        self._set_working(False)
        self._show_search_results(query, transcript)

    def _show_search_results(self, query: str, transcript: Transcript) -> None:
        hits = find_word_occurrences(transcript, query)
        if not hits:
            self._log(f'"{query}": no occurrences found.')
            return
        times = ", ".join(format_timestamp(w.start) for w in hits)
        self._log(f'"{query}" — {len(hits)} occurrence(s) at: {times}')

    def destroy(self) -> None:
        stop_preview()
        super().destroy()
