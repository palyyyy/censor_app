# CensorApp

A desktop application that detects and censors spoken words in audio using speech-to-text. It works on pre-recorded files and on a live microphone stream.

## Overview

CensorApp transcribes speech with a Whisper-based engine, matches the recognized words against a user-defined list, and replaces any matches in the audio. Each word in the list can be censored with a beep tone, with silence, or with a custom sound effect.

There are two modes:

- **File mode** — load an `.mp3` or `.wav` file and export a censored audio file plus a `.txt` transcript that marks which words were censored.
- **Live mode** — route a microphone through the app to a chosen output device. Banned words are replaced in real time using a short lookahead delay.

## Features

- Two speech-to-text backends: faster-whisper (default, runs on Windows/Linux/macOS) and MLX Whisper (Apple Silicon). The active engine is chosen in Settings.
- Per-word replacement mode: beep, silence, or a custom SFX file.
- Word-level timestamps from the STT engine are used to splice replacements at the right position.
- Configurable lookahead delay in live mode (default 2 s) to trade latency against detection reliability.
- Input/output device selection, including virtual devices.
- Light/dark interface built with CustomTkinter.

## Requirements

- Python 3.10 or newer
- `ffmpeg` installed on the system (needed for MP3 input/output)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

## Installation

```bash
cd censor_app
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Apple Silicon, the MLX backend is optional and installed separately:

```bash
pip install mlx mlx-whisper
```

## Usage

```bash
python main.py
```

The main window has two options. **File mode** lets you pick an audio file, build a word list, run the censoring, and save the result. **Live mode** lets you pick input and output devices and start real-time censoring. The first run downloads the selected Whisper model, which can take a moment.

## Project structure

```
censor_app/
├── main.py              entry point
├── config.py            settings and constants
├── requirements.txt
└── app/
    ├── stt/             speech-to-text engines
    │   ├── base.py              engine interface, Word and Transcript types
    │   ├── registry.py          engine registration and lookup
    │   ├── faster_whisper_engine.py
    │   └── mlx_whisper_engine.py
    ├── censor/          matching and replacement logic
    │   ├── censor_rules.py      word list and censor modes
    │   ├── word_matcher.py      word matching
    │   └── audio_processor.py   applies censors to an audio buffer
    ├── audio/
    │   ├── effects.py           beep / silence / SFX generators
    │   ├── device_manager.py    audio device enumeration
    │   ├── file_processor.py    file-mode pipeline
    │   └── live_processor.py    live ring buffer and STT thread
    └── gui/
        ├── main_window.py
        ├── file_mode_window.py
        ├── live_mode_window.py
        ├── settings_window.py
        └── components.py        shared word-list editor
```

## How it works

In file mode, the audio is loaded as mono, transcribed in full, and each word with word-level timing is checked against the word list. Matching regions are replaced with the chosen effect and the file is written back out.

In live mode, incoming microphone samples are written to a ring buffer. A background thread transcribes recent audio in short chunks and flags any matching words. The output stream plays back from the buffer at a fixed delay, and replacements are spliced in just before the audio reaches the speaker. The lookahead delay controls how much time the system has to detect a word before it is played, so a larger delay improves reliability at the cost of latency.

## Configuration

Settings are stored in `~/.censor_app/settings.json` and can be changed from the Settings window: STT engine, model, language, live lookahead, and appearance.

## Notes

- Live mode transcribes short audio chunks, so accuracy is lower than file mode, where the whole file is transcribed at once.
- A larger Whisper model improves recognition but increases processing time.