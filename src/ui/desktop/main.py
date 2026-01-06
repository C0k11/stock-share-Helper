import asyncio
import argparse
import contextlib
import http.server
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import edge_tts
import pygame
import requests
import yaml
from PySide6.QtCore import QObject, Qt, QEvent, QPoint, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QColor, QCursor
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea, QSplitter, QStackedLayout, QVBoxLayout, QWidget, QInputDialog, QPlainTextEdit
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView


_GPT_SOVITS_REF_CACHE: dict[tuple[str, ...], list[dict[str, str]]] = {}
_GPT_SOVITS_REF_CACHE_LOCK = threading.Lock()


class TTSThread(QThread):
    finished_signal = Signal(str)
    started_signal = Signal()  # Emitted when audio starts playing

    def __init__(
        self,
        text: str,
        *,
        voice: str = "zh-CN-XiaoxiaoNeural",
        backend: str = "edge",
        gpt_sovits: dict | None = None,
        preset: str = "gentle",
    ):
        super().__init__()
        self.text = self._normalize_tts_text(str(text))
        self.voice = str(voice)
        self.backend = str(backend or "edge").strip() or "edge"
        self.gpt_sovits = dict(gpt_sovits) if isinstance(gpt_sovits, dict) else {}
        self.preset = str(preset or "gentle").strip() or "gentle"

    @staticmethod
    def _digits_to_zh(s: str) -> str:
        digits = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        }

        def int_to_zh(n: int) -> str:
            if n == 0:
                return "零"
            units = ["", "十", "百", "千"]
            out: list[str] = []
            zero_pending = False
            for i in range(3, -1, -1):
                d = (n // (10 ** i)) % 10
                if d == 0:
                    if out:
                        zero_pending = True
                    continue
                if zero_pending:
                    out.append("零")
                    zero_pending = False
                if i == 1 and d == 1 and not out:
                    out.append("十")
                else:
                    out.append(digits[str(d)] + units[i])
            return "".join(out)

        def big_int_to_zh(n: int) -> str:
            if n == 0:
                return "零"
            big_units = ["", "万", "亿", "兆"]
            parts: list[tuple[int, int]] = []
            u = 0
            x = n
            while x > 0 and u < len(big_units):
                parts.append((x % 10000, u))
                x //= 10000
                u += 1

            res: list[str] = []
            zero_pending = False
            for grp, unit_idx in reversed(parts):
                if grp == 0:
                    zero_pending = True
                    continue
                if zero_pending and res:
                    res.append("零")
                zero_pending = False
                chunk = int_to_zh(grp)
                if chunk != "零":
                    chunk = chunk + big_units[unit_idx]
                res.append(chunk)

            s0 = "".join(res).rstrip("零")
            return s0 or "零"

        def repl(m: re.Match) -> str:
            raw = str(m.group(0) or "")
            raw = raw.replace(",", "")

            sign = ""
            if raw.startswith("-"):
                sign = "负"
                raw = raw[1:]
            elif raw.startswith("+"):
                sign = "加"
                raw = raw[1:]

            is_percent = raw.endswith("%")
            if is_percent:
                raw = raw[:-1]

            if not raw:
                return m.group(0)

            if "." in raw:
                a, b = raw.split(".", 1)
            else:
                a, b = raw, ""

            try:
                n = int(a) if a else 0
            except Exception:
                n = None

            if n is None:
                int_part = "".join(digits.get(ch, ch) for ch in a)
            else:
                int_part = big_int_to_zh(abs(n))

            frac_part = ""
            if b:
                frac_part = "点" + "".join(digits.get(ch, ch) for ch in b)

            body = int_part + frac_part
            if is_percent:
                body = "百分之" + body
            return sign + body

        return re.sub(r"[+-]?[0-9][0-9,]*(?:\.[0-9]+)?%?", repl, str(s or ""))

    @staticmethod
    def _abbr_to_ja_letters(s: str) -> str:
        letter_map = {
            "A": "エー",
            "B": "ビー",
            "C": "シー",
            "D": "ディー",
            "E": "イー",
            "F": "エフ",
            "G": "ジー",
            "H": "エイチ",
            "I": "アイ",
            "J": "ジェー",
            "K": "ケー",
            "L": "エル",
            "M": "エム",
            "N": "エヌ",
            "O": "オー",
            "P": "ピー",
            "Q": "キュー",
            "R": "アール",
            "S": "エス",
            "T": "ティー",
            "U": "ユー",
            "V": "ヴィー",
            "W": "ダブリュー",
            "X": "エックス",
            "Y": "ワイ",
            "Z": "ゼット",
        }

        def repl(m: re.Match) -> str:
            w = str(m.group(0) or "").strip()
            if not w:
                return w
            parts = [letter_map.get(ch, ch) for ch in w]
            return "・".join(parts)

        return re.sub(r"\\b[A-Z]{2,6}\\b", repl, str(s or ""))

    @staticmethod
    def _normalize_tts_text(s: str) -> str:
        x = str(s or "")
        x = TTSThread._digits_to_zh(x)
        return x

    def _play_audio_file(self, path: str) -> Optional[str]:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            self.started_signal.emit()  # Signal that audio started
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(20)
            pygame.mixer.quit()
            return None
        except Exception as e:
            return f"Audio Error: {e}"

    @staticmethod
    def _read_text_file(path: Path) -> str:
        raw: bytes
        try:
            raw = path.read_bytes()
        except Exception:
            return ""

        s = ""
        for enc in ("utf-8", "utf-8-sig", "cp932", "shift_jis", "utf-16", "utf-16le", "utf-16be"):
            try:
                s = raw.decode(enc)
                break
            except Exception:
                continue
        if not s:
            try:
                s = raw.decode("utf-8", errors="ignore")
            except Exception:
                return ""
        return " ".join(str(s).strip().split())

    @staticmethod
    def _load_ref_pool(ref_dirs: list[str]) -> list[dict[str, str]]:
        dirs = [str(d or "").strip() for d in (ref_dirs or []) if str(d or "").strip()]
        key = tuple(dirs)
        if not key:
            return []

        with _GPT_SOVITS_REF_CACHE_LOCK:
            cached = _GPT_SOVITS_REF_CACHE.get(key)
            if isinstance(cached, list) and cached:
                return cached

        audio_exts = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}
        items: list[dict[str, str]] = []
        for d in dirs:
            dp = Path(d)
            if not dp.exists() or not dp.is_dir():
                continue
            try:
                files = list(dp.iterdir())
            except Exception:
                continue

            for fp in files:
                try:
                    if not fp.is_file():
                        continue
                except Exception:
                    continue
                if fp.suffix.lower() not in audio_exts:
                    continue

                txt1 = fp.with_name(fp.name + ".txt")
                txt2 = fp.with_name(fp.stem + ".txt")
                prompt = ""
                if txt1.exists() and txt1.is_file():
                    prompt = TTSThread._read_text_file(txt1)
                elif txt2.exists() and txt2.is_file():
                    prompt = TTSThread._read_text_file(txt2)

                items.append({"audio": str(fp), "prompt_text": str(prompt)})

        random.shuffle(items)
        with _GPT_SOVITS_REF_CACHE_LOCK:
            _GPT_SOVITS_REF_CACHE[key] = items
        return items

    @staticmethod
    def _extract_gpt_sovits_error(resp: requests.Response) -> str:
        try:
            obj = resp.json()
        except Exception:
            return str(resp.text or "").strip()[:300]
        if isinstance(obj, dict):
            msg = obj.get("message") or obj.get("detail") or obj.get("error")
            ex = obj.get("Exception") or obj.get("exception")
            s = " ".join([str(x) for x in [msg, ex] if isinstance(x, str) and x.strip()]).strip()
            return s[:300]
        return str(resp.text or "").strip()[:300]

    def _run_edge_tts(self) -> Optional[str]:
        out_path = os.path.join(tempfile.gettempdir(), "seraphina_tts.mp3")

        async def _gen() -> None:
            communicate = edge_tts.Communicate(self.text, self.voice)
            await communicate.save(out_path)

        try:
            asyncio.run(_gen())
        except Exception:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_gen())
                loop.close()
            except Exception as e:
                return f"TTS Error: {e}"

        err = self._play_audio_file(out_path)
        if err:
            return err
        return None

    def _decode_gpt_sovits_response(self, resp: requests.Response) -> Optional[bytes]:
        ct = str(resp.headers.get("content-type") or "").lower()
        if ct.startswith("audio/"):
            return resp.content

        try:
            obj = resp.json()
        except Exception:
            return None

        if isinstance(obj, dict):
            b64 = obj.get("audio") or obj.get("data") or obj.get("wav")
            if isinstance(b64, str) and b64.strip():
                try:
                    return base64.b64decode(b64)
                except Exception:
                    return None
        return None

    def _run_gpt_sovits(self) -> Optional[str]:
        api_base = str(self.gpt_sovits.get("api_base") or "").strip()
        endpoint = str(self.gpt_sovits.get("endpoint") or "/").strip() or "/"
        text_language = str(self.gpt_sovits.get("text_language") or "zh").strip() or "zh"
        prompt_language = str(self.gpt_sovits.get("prompt_language") or "ja").strip() or "ja"
        gpt_path = str(self.gpt_sovits.get("gpt_path") or "").strip()
        sovits_path = str(self.gpt_sovits.get("sovits_path") or "").strip()
        presets = self.gpt_sovits.get("presets") if isinstance(self.gpt_sovits.get("presets"), dict) else {}
        preset_obj = presets.get(self.preset) if isinstance(presets.get(self.preset), dict) else {}
        refer_wav_path = str((preset_obj or {}).get("refer_wav_path") or "").strip()
        prompt_text = str((preset_obj or {}).get("prompt_text") or "").strip()
        ref_dirs = (preset_obj or {}).get("ref_dirs")
        if not isinstance(ref_dirs, list):
            ref_dirs = []

        if not api_base:
            return "GPT-SoVITS Error: api_base missing"

        url = api_base.rstrip("/") + "/" + endpoint.lstrip("/")
        timeout_sec = int(self.gpt_sovits.get("timeout_sec") or 120)
        max_ref_tries = int(self.gpt_sovits.get("max_ref_tries") or 6)

        pool = self._load_ref_pool(ref_dirs) if ref_dirs else []
        tried: set[str] = set()
        last_err: Optional[str] = None

        for _ in range(max(1, max_ref_tries)):
            picked_audio = refer_wav_path
            picked_prompt = prompt_text
            if pool:
                cand = None
                for it in pool:
                    a = str(it.get("audio") or "").strip()
                    if a and a not in tried:
                        cand = it
                        break
                if cand is None:
                    tried.clear()
                    cand = pool[0]
                picked_audio = str(cand.get("audio") or "").strip()
                tried.add(picked_audio)
                cand_prompt = str(cand.get("prompt_text") or "").strip()
                if cand_prompt:
                    picked_prompt = cand_prompt

            if not picked_audio:
                return "GPT-SoVITS Error: refer_wav_path missing"

            payload = {
                "text": self.text,
                "text_lang": text_language,
                "ref_audio_path": picked_audio,
                "prompt_lang": prompt_language,
                "prompt_text": picked_prompt,
            }
            if gpt_path:
                payload["gpt_path"] = gpt_path
            if sovits_path:
                payload["sovits_path"] = sovits_path

            try:
                resp = requests.post(url, json=payload, timeout=timeout_sec)
            except Exception as e:
                last_err = f"GPT-SoVITS Error: {e}"
                continue

            if not resp.ok:
                detail = self._extract_gpt_sovits_error(resp)
                last_err = f"GPT-SoVITS Error: HTTP {resp.status_code} {detail}".strip()
                continue

            audio_bytes = self._decode_gpt_sovits_response(resp)
            if not audio_bytes:
                last_err = "GPT-SoVITS Error: bad response"
                continue

            out_path = os.path.join(tempfile.gettempdir(), "seraphina_tts.wav")
            try:
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
            except Exception as e:
                return f"GPT-SoVITS Error: {e}"

            err = self._play_audio_file(out_path)
            if err:
                return err
            return None

        return last_err or "GPT-SoVITS Error: failed"

    def run(self) -> None:
        backend = self.backend.lower().strip()
        err: Optional[str] = None
        if backend in {"gpt_sovits", "gpt-sovits", "sovits"}:
            err = self._run_gpt_sovits()
            if err is not None:
                fallback = bool(self.gpt_sovits.get("fallback_to_edge", True))
                if fallback:
                    err2 = self._run_edge_tts()
                    if err2 is None:
                        self.finished_signal.emit("")
                        return
                self.finished_signal.emit(err)
                return

            self.finished_signal.emit("")
            return

        err = self._run_edge_tts()
        if err is not None:
            self.finished_signal.emit(err)
            return
        self.finished_signal.emit("")


class BackendBridge(QObject):
    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window

    @Slot(str)
    def log(self, message: str) -> None:
        print(f"[JS -> Py] {message}")


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        control_tower_url: str,
        start_services: bool,
        avatar_model: str | None,
        api_host: str = "127.0.0.1",
        api_port: int = 8000,
        ui_host: str = "127.0.0.1",
        ui_port: int = 8501,
    ) -> None:
        super().__init__()
        self.setWindowTitle("AI 玛丽交易秘书")
        self.resize(1280, 820)

        self._theme = "dark"
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        self._gaze_dirty = False
        self._gaze_last_x = 0.0
        self._gaze_last_y = 0.0
        self._avatar_ready = False
        self._splitter: QSplitter | None = None
        self._stage_placeholder: QWidget | None = None
        self._stage_spacer: QWidget | None = None
        self._v_splitter: QSplitter | None = None
        self._bottom_splitter: QSplitter | None = None
        self._chat_panel_layout: QVBoxLayout | None = None

        self._api_proc: subprocess.Popen | None = None
        self._ui_proc: subprocess.Popen | None = None
        self._web_server: http.server.ThreadingHTTPServer | None = None
        self._web_thread: threading.Thread | None = None
        self._web_port: int | None = None
        self._session_id = "desktop_" + uuid.uuid4().hex[:12]
        self._api_base = f"http://{api_host}:{int(api_port)}"
        self._ui_base = f"http://{ui_host}:{int(ui_port)}"
        self._control_tower_url = str(control_tower_url)
        self._tts_voice = "zh-CN-XiaoxiaoNeural"
        self._tts_backend = "edge"
        self._gpt_sovits_cfg: dict = {}
        self._auto_execute_actions = False
        self._show_action_notes = False
        self._dashboard_ready = False
        self._pending_dashboard_actions: list[dict] = []

        try:
            cfg_path = Path(self._repo_root()) / "configs" / "secretary.yaml"
            if cfg_path.exists():
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                if isinstance(cfg, dict):
                    voice_cfg = cfg.get("voice") if isinstance(cfg.get("voice"), dict) else {}
                    v = str((voice_cfg or {}).get("tts_voice") or "").strip()
                    if v:
                        self._tts_voice = v
                    b = str((voice_cfg or {}).get("backend") or "").strip()
                    if b:
                        self._tts_backend = b
                    g = (voice_cfg or {}).get("gpt_sovits")
                    if isinstance(g, dict):
                        self._gpt_sovits_cfg = dict(g)
                    actions_cfg = cfg.get("actions") if isinstance(cfg.get("actions"), dict) else {}
                    self._auto_execute_actions = bool(actions_cfg.get("auto_execute", False))
                    self._show_action_notes = bool(actions_cfg.get("show_notes", False))
                    print(f"[TTS Config] backend={self._tts_backend}, voice={self._tts_voice}")
                    print(f"[TTS Config] gpt_sovits fallback_to_edge={self._gpt_sovits_cfg.get('fallback_to_edge')}")
        except Exception as e:
            print(f"[TTS Config] Error loading config: {e}")

        if start_services:
            self._start_services(api_host=api_host, api_port=api_port, ui_host=ui_host, ui_port=ui_port)

        self.dashboard_view = QWebEngineView()
        self.avatar_view = QWebEngineView()
        self.avatar_view.page().setBackgroundColor(QColor(0, 0, 0, 0))
        self.avatar_view.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.avatar_view.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        self.channel = QWebChannel()
        self.bridge = BackendBridge(self)
        self.channel.registerObject("backend", self.bridge)
        self.avatar_view.page().setWebChannel(self.channel)

        avatar_url = self._start_avatar_server(avatar_model=avatar_model)
        self.avatar_view.setUrl(QUrl(avatar_url))

        dash_url = f"http://127.0.0.1:{int(self._web_port)}/dashboard.html?api={urllib.parse.quote(self._api_base + '/api/v1', safe='')}" if self._web_port else ""
        if dash_url:
            self.dashboard_view.setUrl(QUrl(dash_url))

        # Only call JS APIs after the page defines them.
        self.avatar_view.loadFinished.connect(self._on_avatar_loaded)

        self._apply_theme(self._theme)

        toggle = QAction("切换主题", self)
        toggle.triggered.connect(self._toggle_theme)
        self.addAction(toggle)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)


        # Ensure dashboard theming is applied after the page is actually ready.
        self.dashboard_view.loadFinished.connect(lambda _ok: self._apply_theme(self._theme))
        self.dashboard_view.loadFinished.connect(self._on_dashboard_loaded)
        QTimer.singleShot(1500, lambda: self._apply_theme(self._theme))

        # Right chat panel (ChatGPT style) - lives as a normal widget (no overlay).
        chat_panel = QWidget()
        chat_panel.setObjectName("chatPanel")
        self._chat_panel = chat_panel
        chat_panel_layout = QVBoxLayout()
        self._chat_panel_layout = chat_panel_layout
        chat_panel_layout.setContentsMargins(12, 16, 16, 16)
        chat_panel_layout.setSpacing(12)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.chat_scroll.setStyleSheet("background: transparent;")
        self.chat_scroll.setMinimumHeight(360)

        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background: transparent;")
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch(1)
        self.chat_container.setLayout(self.chat_layout)
        self.chat_scroll.setWidget(self.chat_container)

        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("输入消息，回车发送")
        self.input_box.returnPressed.connect(self.send_message)
        btn = QPushButton("发送")
        btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.input_box)
        input_layout.addWidget(btn)

        # Nightly training panel (Ouroboros RLHF)
        train_panel = QFrame()
        train_panel.setObjectName("trainPanel")
        train_panel.setStyleSheet(
            """
            QFrame#trainPanel {
              background: rgba(0, 0, 0, 0.18);
              border: 1px solid rgba(255,255,255,0.10);
              border-radius: 10px;
            }
            """
        )
        train_layout = QVBoxLayout()
        train_layout.setContentsMargins(10, 8, 10, 10)
        train_layout.setSpacing(6)

        train_top = QHBoxLayout()
        train_top.setContentsMargins(0, 0, 0, 0)
        lbl_train = QLabel("Nightly Training")
        lbl_train.setStyleSheet("color: rgba(229,231,235,0.90); font-weight: 800;")
        self._btn_train_start = QPushButton("Start Training")
        self._btn_train_stop = QPushButton("Stop Training")
        self._btn_train_start.setStyleSheet("background: rgba(16,185,129,0.28); border: 1px solid rgba(16,185,129,0.35);")
        self._btn_train_stop.setStyleSheet("background: rgba(239,68,68,0.22); border: 1px solid rgba(239,68,68,0.30);")
        self._lbl_train_status = QLabel("")
        self._lbl_train_status.setStyleSheet("color: rgba(229,231,235,0.72);")
        train_top.addWidget(lbl_train)
        train_top.addStretch(1)
        train_top.addWidget(self._btn_train_start)
        train_top.addWidget(self._btn_train_stop)

        train_layout.addLayout(train_top)
        train_layout.addWidget(self._lbl_train_status)

        self._train_log = QPlainTextEdit()
        self._train_log.setReadOnly(True)
        self._train_log.setStyleSheet(
            """
            QPlainTextEdit {
              background: rgba(0, 0, 0, 0.25);
              color: rgba(229,231,235,0.86);
              border: 1px solid rgba(255,255,255,0.08);
              border-radius: 8px;
              font-family: Consolas, 'Monaco', monospace;
              font-size: 11px;
            }
            """
        )
        self._train_log.setMaximumHeight(180)
        self._train_log.setPlaceholderText("(No training yet)")
        train_layout.addWidget(self._train_log)
        train_panel.setLayout(train_layout)

        self._btn_train_start.clicked.connect(self._start_nightly_training)
        self._btn_train_stop.clicked.connect(self._stop_nightly_training)

        self._train_poll_timer = QTimer()
        self._train_poll_timer.setInterval(2000)
        self._train_poll_timer.timeout.connect(self._poll_nightly_training)
        self._train_poll_timer.start()

        chat_panel_layout.addWidget(self.chat_scroll, stretch=1)
        chat_panel_layout.addWidget(train_panel, stretch=0)
        chat_panel_layout.addLayout(input_layout)
        chat_panel.setLayout(chat_panel_layout)

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.setHandleWidth(1)
        bottom_splitter.setStyleSheet("QSplitter::handle { background: transparent; }")
        try:
            bottom_splitter.setChildrenCollapsible(False)
        except Exception:
            pass
        bottom_splitter.addWidget(self.avatar_view)
        bottom_splitter.addWidget(chat_panel)
        bottom_splitter.setStretchFactor(0, 3)
        bottom_splitter.setStretchFactor(1, 5)
        self._bottom_splitter = bottom_splitter

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.setHandleWidth(1)
        v_splitter.setStyleSheet("QSplitter::handle { background: transparent; }")
        try:
            v_splitter.setChildrenCollapsible(False)
        except Exception:
            pass
        v_splitter.addWidget(self.dashboard_view)
        v_splitter.addWidget(bottom_splitter)
        v_splitter.setStretchFactor(0, 7)
        v_splitter.setStretchFactor(1, 3)
        self._v_splitter = v_splitter

        try:
            self.dashboard_view.setMinimumWidth(720)
            self.avatar_view.setMinimumWidth(320)
            chat_panel.setMinimumWidth(420)
        except Exception:
            pass

        central = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(v_splitter, stretch=1)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.is_talking = False
        self.tts_thread: TTSThread | None = None
        self._chat_thread: QThread | None = None

        self.mouth_timer = QTimer()
        self.mouth_timer.timeout.connect(self._update_mouth)

        self._gaze_timer = QTimer()
        self._gaze_timer.setInterval(33)
        self._gaze_timer.timeout.connect(self._flush_gaze)
        self._gaze_timer.start()

        QApplication.instance().installEventFilter(self)

        self._enable_mouse_tracking(self)

        self._append_chat(role="assistant", text="系统已就绪。")
        QTimer.singleShot(800, self._push_dock_hint)

        QTimer.singleShot(260, self._apply_initial_splitter_sizes)

        try:
            v_splitter.splitterMoved.connect(lambda _pos, _idx: self._push_dock_hint())
            bottom_splitter.splitterMoved.connect(lambda _pos, _idx: self._push_dock_hint())
        except Exception:
            pass

        # Initial poll (after UI is fully constructed)
        QTimer.singleShot(1200, self._poll_nightly_training)

    def _post_nightly_train_start(self) -> dict:
        url = f"{self._api_base}/api/v1/evolution/nightly/train/start"
        resp = requests.post(url, timeout=30)
        obj: Any = resp.json() if resp.status_code == 200 else {"ok": False, "error": resp.text}
        return obj if isinstance(obj, dict) else {"ok": False, "error": "bad response"}

    def _post_nightly_train_stop(self) -> dict:
        url = f"{self._api_base}/api/v1/evolution/nightly/train/stop"
        resp = requests.post(url, timeout=15)
        obj: Any = resp.json() if resp.status_code == 200 else {"ok": False, "error": resp.text}
        return obj if isinstance(obj, dict) else {"ok": False, "error": "bad response"}

    def _get_nightly_train_status(self) -> dict:
        url = f"{self._api_base}/api/v1/evolution/nightly/train/status?tail_bytes=16000"
        resp = requests.get(url, timeout=6)
        obj: Any = resp.json() if resp.status_code == 200 else {"ok": False, "error": resp.text}
        return obj if isinstance(obj, dict) else {"ok": False, "error": "bad response"}

    def _set_train_ui_state(self, *, running: bool, status_text: str, log_tail: str, next_adapter: str = "") -> None:
        try:
            self._btn_train_start.setEnabled(not running)
            self._btn_train_stop.setEnabled(bool(running))
        except Exception:
            pass
        try:
            s = str(status_text or "").strip()
            if next_adapter:
                s = (s + f" | next adapter: {next_adapter}").strip()
            self._lbl_train_status.setText(s)
        except Exception:
            pass
        try:
            txt = str(log_tail or "").strip()
            if not txt:
                txt = "(No training yet)"
            self._train_log.setPlainText(txt)
            sb = self._train_log.verticalScrollBar()
            sb.setValue(sb.maximum())
        except Exception:
            pass

    def _start_nightly_training(self) -> None:
        try:
            self._btn_train_start.setEnabled(False)
        except Exception:
            pass

        class _TrainStartThread(QThread):
            finished = Signal(object)

            def __init__(self, outer: "MainWindow"):
                super().__init__()
                self.outer = outer

            def run(self) -> None:
                try:
                    r = self.outer._post_nightly_train_start()
                except Exception as e:
                    r = {"ok": False, "error": str(e)}
                self.finished.emit(r)

        def _done(res: Any) -> None:
            ok = bool(isinstance(res, dict) and res.get("ok"))
            if not ok:
                err = str(res.get("error") if isinstance(res, dict) else res)
                self._set_train_ui_state(running=False, status_text=f"train start failed: {err}", log_tail="")
            self._poll_nightly_training()

        t = _TrainStartThread(self)
        t.finished.connect(_done)
        t.start()
        self._train_start_thread = t

    def _stop_nightly_training(self) -> None:
        try:
            self._btn_train_stop.setEnabled(False)
        except Exception:
            pass

        class _TrainStopThread(QThread):
            finished = Signal(object)

            def __init__(self, outer: "MainWindow"):
                super().__init__()
                self.outer = outer

            def run(self) -> None:
                try:
                    r = self.outer._post_nightly_train_stop()
                except Exception as e:
                    r = {"ok": False, "error": str(e)}
                self.finished.emit(r)

        def _done(_res: Any) -> None:
            self._poll_nightly_training()

        t = _TrainStopThread(self)
        t.finished.connect(_done)
        t.start()
        self._train_stop_thread = t

    def _poll_nightly_training(self) -> None:
        # Avoid piling up threads if the API is slow.
        try:
            th = getattr(self, "_train_status_thread", None)
            if isinstance(th, QThread) and th.isRunning():
                return
        except Exception:
            pass

        class _TrainStatusThread(QThread):
            finished = Signal(object)

            def __init__(self, outer: "MainWindow"):
                super().__init__()
                self.outer = outer

            def run(self) -> None:
                try:
                    r = self.outer._get_nightly_train_status()
                except Exception as e:
                    r = {"ok": False, "error": str(e)}
                self.finished.emit(r)

        def _done(res: Any) -> None:
            if not isinstance(res, dict) or not res.get("ok"):
                err = str(res.get("error") if isinstance(res, dict) else res)
                self._set_train_ui_state(running=False, status_text=f"train status error: {err}", log_tail="")
                return

            running = bool(res.get("running"))
            pid = str(res.get("pid") or "").strip()
            rc = res.get("returncode")
            tail = str(res.get("log_tail") or "")

            next_adapter = ""
            try:
                meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
                outs = meta.get("outputs") if isinstance(meta.get("outputs"), dict) else {}
                next_adapter = str(outs.get("next_dpo_adapter") or "").strip()
            except Exception:
                next_adapter = ""

            if running:
                st = "Training running" + (f" (pid={pid})" if pid else "")
            else:
                st = "Training idle"
                if rc is not None:
                    st = f"Training finished (rc={rc})"
            self._set_train_ui_state(running=running, status_text=st, log_tail=tail, next_adapter=next_adapter)

        t = _TrainStatusThread(self)
        t.finished.connect(_done)
        t.start()
        self._train_status_thread = t

    def _extract_action_blocks(self, text: str) -> tuple[str, list[dict]]:
        t = str(text or "")
        actions: list[dict] = []
        pattern = re.compile(r"```action\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)

        def _repl(m: re.Match) -> str:
            raw = m.group(1)
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and isinstance(obj.get("action"), str):
                    actions.append(obj)
            except Exception:
                pass
            return ""

        cleaned = pattern.sub(_repl, t)
        cleaned = cleaned.strip()
        return cleaned, actions

    def _post_action(self, action_obj: dict) -> dict:
        url = f"{self._api_base}/api/v1/actions/execute"
        data = json.dumps(action_obj, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        obj: Any = json.loads(raw)
        return obj if isinstance(obj, dict) else {"ok": False, "error": "bad response"}

    def _on_dashboard_loaded(self, ok: bool) -> None:
        self._dashboard_ready = bool(ok)
        if not self._dashboard_ready:
            return
        if self._pending_dashboard_actions:
            pending = list(self._pending_dashboard_actions)
            self._pending_dashboard_actions.clear()
            for a in pending:
                try:
                    self._execute_ui_action(a)
                except Exception:
                    pass

    def _run_dashboard_js(self, js: str, *, label: str) -> None:
        try:
            page = self.dashboard_view.page()
        except Exception:
            return

        def _cb(res: Any) -> None:
            try:
                print(f"[UIAction] {label} -> {res}")
            except Exception:
                return

        try:
            page.runJavaScript(js, _cb)
        except Exception:
            try:
                page.runJavaScript(js)
            except Exception:
                pass

    def _execute_ui_action(self, action_obj: dict) -> dict:
        a = str((action_obj or {}).get("action") or "").strip().lower()
        params = (action_obj or {}).get("params") if isinstance((action_obj or {}).get("params"), dict) else {}
        if not hasattr(self, "dashboard_view") or self.dashboard_view is None:
            return {"ok": False, "error": "dashboard_view not ready"}

        if not getattr(self, "_dashboard_ready", False):
            self._pending_dashboard_actions.append(dict(action_obj or {}))
            return {"ok": True, "queued": True}

        if a == "ui.refresh":
            self._run_dashboard_js(
                "(function(){try{if(window.loadLiveData){window.loadLiveData();return {ok:true,called:true};}return {ok:false,called:false};}catch(e){return {ok:false,error:String(e)};}})();",
                label="ui.refresh",
            )
            return {"ok": True}

        if a == "ui.set_live_ticker":
            tk = str(params.get("ticker") or params.get("symbol") or "").strip().upper()
            if not tk:
                return {"ok": False, "error": "ticker required"}
            js = (
                "(function(){try{"
                "var sel=document.getElementById('liveTickerSelect');"
                f"if(sel){{var v={json.dumps(tk)};var ok=false;for(var i=0;i<sel.options.length;i++){{if(sel.options[i].value===v){{ok=true;break;}}}}if(!ok){{sel.add(new Option(v,v));}}sel.value=v;}}"
                "if (window.loadLiveChart){window.loadLiveChart();}"
                "return {ok:true,ticker:(sel&&sel.value)||null,hasSel:!!sel,hasLoad:(typeof window.loadLiveChart==='function')};"
                "}catch(e){return {ok:false,error:String(e)};}})();"
            )
            self._run_dashboard_js(js, label=f"ui.set_live_ticker:{tk}")
            return {"ok": True, "ticker": tk}

        if a == "ui.set_mode":
            mode = str(params.get("mode") or "").strip().lower()
            if mode not in {"online", "offline"}:
                return {"ok": False, "error": "mode must be online/offline"}
            self.dashboard_view.page().runJavaScript(
                f"if (window.setTradingMode) window.setTradingMode('{mode}');"
            )
            return {"ok": True, "mode": mode}

        return {"ok": False, "error": f"unknown ui action: {a}"}

    def _apply_initial_splitter_sizes(self) -> None:
        try:
            if hasattr(self, "_v_splitter") and self._v_splitter is not None:
                h = max(1, int(self.height()))
                top_h = int(max(240, h * 0.60))
                bottom_h = max(200, h - top_h)
                self._v_splitter.setSizes([top_h, bottom_h])

            if hasattr(self, "_bottom_splitter") and self._bottom_splitter is not None:
                w = max(1, int(self.width()))
                left_w = int(max(320, min(860, w * 0.52)))
                right_w = max(360, w - left_w)
                self._bottom_splitter.setSizes([left_w, right_w])
        except Exception:
            return

        QTimer.singleShot(0, self._push_dock_hint)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        try:
            super().resizeEvent(event)
        except Exception:
            pass
        QTimer.singleShot(0, self._push_dock_hint)

    def _push_dock_hint(self) -> None:
        if not getattr(self, "_avatar_ready", False):
            return
        bottom_pad = 2.0
        try:
            h = max(1, int(self.avatar_view.height()))
        except Exception:
            h = max(1, int(self.height()))
        overflow = min(float(h) * 4.20, 5200.0)
        try:
            top_m = 16
            if hasattr(self, "_chat_panel_layout") and self._chat_panel_layout is not None:
                try:
                    top_m = int(self._chat_panel_layout.contentsMargins().top())
                except Exception:
                    top_m = 16
        except Exception:
            top_m = 16

        # Keep head close to the horizontal splitter (top of the bottom area).
        # We still pass a value derived from layout for future tweaks, but clamp it to be near the top.
        head_top_px = float(max(0, min(10, top_m - 8)))
        self.avatar_view.page().runJavaScript(
            f"if (window.setSceneHint) window.setSceneHint(0, {overflow:.1f}, {bottom_pad:.1f}, 0, 0, 0.0, {head_top_px:.1f});"
        )

    @staticmethod
    def _enable_mouse_tracking(w: QWidget) -> None:
        try:
            w.setMouseTracking(True)
        except Exception:
            return
        for c in w.findChildren(QWidget):
            try:
                c.setMouseTracking(True)
            except Exception:
                pass

    def eventFilter(self, obj, event):  # type: ignore[override]
        if event.type() == QEvent.Type.MouseMove:
            try:
                p = event.globalPosition().toPoint()
            except Exception:
                try:
                    p = event.globalPos()
                except Exception:
                    return False

            geo = self.frameGeometry()
            w = max(1, int(geo.width()))
            h = max(1, int(geo.height()))
            rel_x = (p.x() - geo.left()) / float(w)
            rel_y = (p.y() - geo.top()) / float(h)
            rel_x = max(0.0, min(1.0, rel_x))
            rel_y = max(0.0, min(1.0, rel_y))

            x = (rel_x - 0.5) * 2.0
            y = (0.5 - rel_y) * 2.0
            self._gaze_x = max(-1.0, min(1.0, float(x)))
            self._gaze_y = max(-1.0, min(1.0, float(y)))
            self._gaze_dirty = True

        return False

    def _flush_gaze(self) -> None:
        if not self._avatar_ready:
            return
        # Poll global cursor position to make gaze tracking work across the whole window
        # (QWebEngineView may swallow mouse move events).
        try:
            p = QCursor.pos()
        except Exception:
            return

        try:
            # Send mouse position in avatar_view-local pixels; JS will map it to the eye anchor.
            av_local = self.avatar_view.mapFromGlobal(p)
            mx = float(av_local.x())
            my = float(av_local.y())
        except Exception:
            return

        gx = mx
        gy = my

        if abs(gx - self._gaze_last_x) < 0.003 and abs(gy - self._gaze_last_y) < 0.003:
            return

        self._gaze_last_x = gx
        self._gaze_last_y = gy
        self.avatar_view.page().runJavaScript(
            "if (window.setMousePx) window.setMousePx(" +
            f"{gx:.1f}, {gy:.1f}" +
            "); else if (window.setEye) window.setEye(" +
            f"{(0.0):.4f}, {(0.0):.4f}" +
            ");"
        )

    def _toggle_theme(self) -> None:
        self._theme = "light" if self._theme == "dark" else "dark"
        self._apply_theme(self._theme)

    def _apply_theme(self, theme: str) -> None:
        t = "light" if str(theme).lower() == "light" else "dark"
        if t == "light":
            self.setStyleSheet(
                """
                QMainWindow { background: #f8fafc; }
                QLineEdit { background: white; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 10px; padding: 8px 10px; }
                QPushButton { background: #2563eb; color: white; border: none; border-radius: 10px; padding: 8px 14px; }
                QPushButton:hover { background: #1d4ed8; }
                QScrollArea { background: transparent; }
                QWidget#chatPanel { background: rgba(255,255,255,1.0); border: 1px solid rgba(15,23,42,0.10); border-radius: 14px; }
                """
            )
        else:
            self.setStyleSheet(
                """
                QMainWindow { background: #0b0f17; }
                QLineEdit { background: rgba(17,24,39,0.95); color: rgba(255,255,255,0.92); border: 1px solid rgba(255,255,255,0.10); border-radius: 10px; padding: 8px 10px; }
                QPushButton { background: rgba(37,99,235,0.92); color: white; border: none; border-radius: 10px; padding: 8px 14px; }
                QPushButton:hover { background: rgba(29,78,216,0.95); }
                QScrollArea { background: transparent; }
                QWidget#chatPanel { background: rgba(17,24,39,1.0); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; }
                """
            )

        if self._avatar_ready:
            self.avatar_view.page().runJavaScript(f"if (window.setTheme) window.setTheme('{t}');")

        # Best-effort: theme Streamlit page via injected CSS.
        bg = "#f7f7f8" if t == "light" else "#0b0f17"
        fg = "#0f172a" if t == "light" else "#e5e7eb"
        card = "#ffffff" if t == "light" else "#111827"
        border = "rgba(15,23,42,0.10)" if t == "light" else "rgba(255,255,255,0.10)"
        subtle = "rgba(15,23,42,0.06)" if t == "light" else "rgba(255,255,255,0.06)"
        strong = "rgba(15,23,42,0.12)" if t == "light" else "rgba(255,255,255,0.10)"
        accent = "#2563eb" if t == "light" else "#60a5fa"
        code_bg = "rgba(15,23,42,0.04)" if t == "light" else "rgba(0,0,0,0.35)"
        css = (
            f":root{{color-scheme:{t};}}"
            f"html,body,.stApp,div[data-testid='stAppViewContainer']{{background:{bg} !important;color:{fg} !important;font-size:18px !important;line-height:1.55 !important;font-family:Segoe UI,Microsoft YaHei,system-ui,-apple-system,BlinkMacSystemFont,Arial,sans-serif !important;}}"
            f"header,div[data-testid='stHeader']{{background:{bg} !important;}}"
            f"section[data-testid='stSidebar']{{background:{card} !important;border-right:1px solid {border} !important;}}"
            f"div[data-testid='stVerticalBlock'],div[data-testid='stContainer']{{color:{fg} !important;}}"
            f"h1,h2,h3,h4,h5,h6,p,span,div{{color:{fg} !important;}}"

            # IMPORTANT: do NOT globally style stMarkdownContainer as a card.
            # It wraps headings/captions and can create huge empty rounded blocks.
            f"div[data-testid='stMetric'],div[data-testid='stExpander']{{background:{card} !important;border:1px solid {border} !important;border-radius:14px !important;padding:14px !important;}}"
            f"div[data-testid='stMetric'] *{{font-size:15px !important;}}"

            f"div[data-testid='stWidgetLabel']>label{{font-weight:800 !important;letter-spacing:0.2px !important;opacity:1 !important;}}"
            f"div[data-testid='stWidgetLabel'] span{{font-weight:800 !important;opacity:1 !important;}}"
            f"div[data-testid='stWidgetLabel']>label,div[data-testid='stWidgetLabel'] span,section[data-testid='stSidebar'] label{{color:{fg} !important;}}"
            f"div[data-testid='stCaptionContainer'],.stCaption{{color:{'rgba(107,114,128,0.95)' if t=='light' else 'rgba(229,231,235,0.90)'} !important;font-size:14px !important;}}"

            f"section[data-testid='stSidebar'] *{{color:{fg} !important;}}"
            f"button[kind],div.stButton>button{{background:{accent} !important;color:white !important;border:0 !important;border-radius:14px !important;padding:16px 22px !important;font-size:15px !important;font-weight:800 !important;min-height:64px !important;}}"
            f"button[kind]:hover,div.stButton>button:hover{{filter:brightness(1.08);}}"
            f"input,textarea{{background:{card} !important;color:{fg} !important;border:1px solid {strong} !important;border-radius:14px !important;padding:16px 18px !important;font-size:15px !important;min-height:64px !important;line-height:1.20 !important;}}"
            f"div[data-baseweb='select']>div{{background:{card} !important;color:{fg} !important;border:1px solid {strong} !important;border-radius:14px !important;min-height:64px !important;}}"
            # Prevent BaseWeb internal search input from looking like a separate textbox.
            f"div[data-baseweb='select'] input{{color:{fg} !important;-webkit-text-fill-color:{fg} !important;font-size:15px !important;font-family:inherit !important;opacity:1 !important;line-height:1.20 !important;background:transparent !important;border:0 !important;box-shadow:none !important;outline:none !important;caret-color:transparent !important;padding:0 !important;margin:0 !important;}}"
            f"div[data-baseweb='select'] svg{{fill:{fg} !important;}}"
            f"div[data-baseweb='select'] [role='combobox']{{width:100% !important;}}"
            f"div[data-baseweb='select'] [role='combobox'] *{{cursor:pointer !important;}}"
            # BaseWeb select internal layout tends to keep value in a smaller inner pill; force full-height flex.
            f"div[data-baseweb='select']>div{{display:flex !important;align-items:center !important;}}"
            f"div[data-baseweb='select']>div *{{min-height:0 !important;}}"
            f"div[data-baseweb='select'] [role='combobox']{{display:flex !important;align-items:center !important;height:100% !important;}}"
            f"div[data-baseweb='select'] [role='combobox'] input{{position:absolute !important;inset:0 !important;width:100% !important;height:100% !important;opacity:0 !important;pointer-events:none !important;}}"
            f"div[data-baseweb='select'] [role='combobox'] [contenteditable='true']{{width:100% !important;min-width:0 !important;outline:none !important;background:transparent !important;border:0 !important;box-shadow:none !important;caret-color:transparent !important;pointer-events:none !important;}}"
            f"div[data-baseweb='select'] [role='combobox'] [contenteditable='true']{{height:100% !important;display:flex !important;align-items:center !important;}}"
            f"div[data-baseweb='select'] [role='combobox'] span{{color:{fg} !important;opacity:1 !important;font-size:15px !important;font-family:inherit !important;line-height:1.20 !important;}}"
            f"div[data-baseweb='select'] [role='combobox'] [data-testid='stTickBar'] *{{color:{fg} !important;}}"
            f"div[role='listbox'] *{{color:{fg} !important;opacity:1 !important;font-size:15px !important;font-family:inherit !important;}}"
            f"div[data-baseweb='popover'] div[role='option']{{padding-top:10px !important;padding-bottom:10px !important;}}"
            f"div[data-baseweb='popover'] *{{background:{card} !important;color:{fg} !important;}}"
            f"div[data-baseweb='radio'] *{{color:{fg} !important;}}"
            f"div[data-baseweb='slider'] *{{color:{fg} !important;font-size:15px !important;}}"
            f"div[data-testid='stTickBar'] *{{color:{fg} !important;}}"

            # Plotly: prevent white backgrounds and make it look like a dark card.
            f"div[data-testid='stPlotlyChart']>div{{background:transparent !important;}}"
            f"div[data-testid='stPlotlyChart']{{background:{card} !important;border:1px solid {border} !important;border-radius:14px !important;padding:10px !important;}}"

            f"code,pre{{background:{code_bg} !important;color:{fg} !important;border:1px solid {subtle} !important;font-size:14px !important;}}"
            f".metric-card{{background:{card} !important;border:1px solid {border} !important;border-radius:14px !important;padding:14px !important;border-left:5px solid {accent} !important;}}"
            f".bull-msg{{background:{'rgba(230,255,250,0.85)' if t=='light' else 'rgba(16,185,129,0.10)'} !important;color:{fg} !important;border:1px solid {subtle} !important;}}"
            f".bear-msg{{background:{'rgba(255,245,245,0.85)' if t=='light' else 'rgba(239,68,68,0.10)'} !important;color:{fg} !important;border:1px solid {subtle} !important;}}"
            f".judge-msg{{background:{card} !important;color:{fg} !important;border:1px solid {border} !important;}}"
            f".small-muted{{color:{'#6b7280' if t=='light' else 'rgba(229,231,235,0.78)'} !important;}}"
        )
        js = (
            "(function(){try{"
            "var id='__ai_term_theme';"
            "var st=document.getElementById(id);"
            "if(!st){st=document.createElement('style');st.id=id;document.head.appendChild(st);}"
            f"st.textContent={json.dumps(css)};"
            "}catch(e){}})();"
        )
        try:
            if hasattr(self, "dashboard_view") and self.dashboard_view is not None:
                self.dashboard_view.page().runJavaScript(js)
        except Exception:
            pass

    @staticmethod
    def _pick_tts_preset(text: str) -> str:
        t = str(text or "")
        tl = t.lower()
        if any(k in t for k in ["亏", "回撤", "止损", "爆仓", "试炼", "下跌", "跌破"]) or any(k in tl for k in ["drawdown", "stop", "loss", "risk", "crash"]):
            return "worry"
        if any(k in t for k in ["盈利", "赚钱", "福报", "上涨", "恭喜", "祝福"]) or any(k in tl for k in ["profit", "pnl", "gain", "win"]):
            return "happy"
        return "gentle"

    def _on_avatar_loaded(self, ok: bool) -> None:
        self._avatar_ready = bool(ok)
        if not self._avatar_ready:
            return
        t = "light" if str(self._theme).lower() == "light" else "dark"
        self.avatar_view.page().runJavaScript(f"if (window.setTheme) window.setTheme('{t}');")
        
        # Mari greeting on startup
        QTimer.singleShot(2000, self._mari_greeting)

    def _mari_greeting(self) -> None:
        """Mari greets Sensei on startup"""
        greeting = "Sensei, 早上好。系统已就绪，随时为您服务。"
        self._pending_reply = greeting
        
        def _on_voice_started() -> None:
            label = self._append_chat(role="assistant", text=greeting)
            safe = greeting.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
            self.avatar_view.page().runJavaScript(f"window.showBubble('{safe}');")
            self.is_talking = True
            self.mouth_timer.start(50)
        
        preset = self._pick_tts_preset(greeting)
        self.tts_thread = TTSThread(
            greeting,
            voice=str(getattr(self, "_tts_voice", "zh-CN-XiaoxiaoNeural")),
            backend=str(getattr(self, "_tts_backend", "edge")),
            gpt_sovits=dict(getattr(self, "_gpt_sovits_cfg", {}) or {}),
            preset=str(preset),
        )
        self.tts_thread.started_signal.connect(_on_voice_started)
        self.tts_thread.finished_signal.connect(self._on_tts_finished)
        self.tts_thread.start()

    def _append_chat(self, *, role: str, text: str, message_id: str | None = None, enable_feedback: bool = False) -> QLabel:
        bubble = QFrame()
        bubble.setObjectName("chatBubble")
        bubble.setStyleSheet(
            """
            QFrame#chatBubble {
              border-radius: 12px;
              padding: 10px 12px;
            }
            """
        )

        label = QLabel(str(text or ""))
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        try:
            label.setProperty("message_id", str(message_id or "").strip())
        except Exception:
            pass

        bubble_layout = QVBoxLayout()
        bubble_layout.setContentsMargins(12, 10, 12, 10)
        bubble_layout.addWidget(label)

        # RLHF feedback buttons (only for assistant bubbles with message_id)
        if role != "user" and enable_feedback:
            btn_row = QHBoxLayout()
            btn_row.setContentsMargins(0, 4, 0, 0)
            btn_row.addStretch(1)

            btn_like = QPushButton("👍")
            btn_like.setFixedSize(30, 28)
            btn_like.setStyleSheet("background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.30);")

            btn_dislike = QPushButton("👎")
            btn_dislike.setFixedSize(30, 28)
            btn_dislike.setStyleSheet("background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.28);")

            # Disable until message_id is known.
            mid0 = str(message_id or "").strip()
            btn_like.setEnabled(bool(mid0))
            btn_dislike.setEnabled(bool(mid0))

            setattr(label, "_btn_like", btn_like)
            setattr(label, "_btn_dislike", btn_dislike)
            setattr(label, "_feedback_sent", False)

            btn_like.clicked.connect(lambda: self._handle_feedback(label=label, score=1))
            btn_dislike.clicked.connect(lambda: self._handle_feedback(label=label, score=-1))

            btn_row.addWidget(btn_like)
            btn_row.addWidget(btn_dislike)
            bubble_layout.addLayout(btn_row)
        bubble.setLayout(bubble_layout)

        if role == "user":
            label.setStyleSheet("color: white;")
            bubble.setStyleSheet(
                """
                QFrame#chatBubble {
                  background: rgba(37, 99, 235, 0.92);
                  color: white;
                  border-radius: 12px;
                }
                """
            )
        else:
            if self._theme == "light":
                label.setStyleSheet("color: #0f172a;")
                bubble.setStyleSheet(
                    """
                    QFrame#chatBubble {
                      background: rgba(255, 255, 255, 0.95);
                      color: #0f172a;
                      border: 1px solid rgba(15, 23, 42, 0.12);
                      border-radius: 12px;
                    }
                    """
                )
            else:
                label.setStyleSheet("color: rgba(255,255,255,0.92);")
                bubble.setStyleSheet(
                    """
                    QFrame#chatBubble {
                      background: rgba(31, 41, 55, 0.92);
                      color: rgba(255,255,255,0.92);
                      border: 1px solid rgba(255,255,255,0.10);
                      border-radius: 12px;
                    }
                    """
                )

        row = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)
        if role == "user":
            row_layout.addStretch(1)
            row_layout.addWidget(bubble, stretch=0)
        else:
            row_layout.addWidget(bubble, stretch=0)
            row_layout.addStretch(1)
        row.setLayout(row_layout)

        stretch_item = self.chat_layout.takeAt(self.chat_layout.count() - 1)
        if stretch_item is not None:
            pass
        self.chat_layout.addWidget(row)
        self.chat_layout.addStretch(1)

        QTimer.singleShot(0, lambda: self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum()))

        return label

    def _post_feedback(self, *, message_id: str, score: int, comment: str = "") -> dict:
        url = f"{self._api_base}/api/v1/feedback"
        payload = {
            "message_id": str(message_id),
            "score": int(score),
            "comment": str(comment or ""),
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=25.0) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        obj: Any = json.loads(raw)
        return obj if isinstance(obj, dict) else {"ok": False, "error": "bad response"}

    def _handle_feedback(self, *, label: QLabel, score: int) -> None:
        try:
            if bool(getattr(label, "_feedback_sent", False)):
                return
        except Exception:
            pass

        try:
            mid = str(label.property("message_id") or "").strip()
        except Exception:
            mid = ""
        if not mid:
            return

        # Optimistic disable to avoid duplicate submits.
        try:
            setattr(label, "_feedback_sent", True)
        except Exception:
            pass

        comment = ""
        if int(score) == -1:
            try:
                text, ok = QInputDialog.getText(self, "纠错", "Sensei，我哪里说错了？请教教我：")
                if ok:
                    comment = str(text or "")
            except Exception:
                comment = ""

        try:
            btn_like = getattr(label, "_btn_like", None)
            btn_dislike = getattr(label, "_btn_dislike", None)
        except Exception:
            btn_like = None
            btn_dislike = None

        try:
            if btn_like is not None:
                btn_like.setEnabled(False)
            if btn_dislike is not None:
                btn_dislike.setEnabled(False)
        except Exception:
            pass

        class _FbThread(QThread):
            finished = Signal(object)

            def __init__(self, outer: "MainWindow", message_id: str, score: int, comment: str):
                super().__init__()
                self.outer = outer
                self.message_id = message_id
                self.score = score
                self.comment = comment

            def run(self) -> None:
                try:
                    r = self.outer._post_feedback(message_id=self.message_id, score=self.score, comment=self.comment)
                except Exception as e:
                    r = {"ok": False, "error": str(e)}
                self.finished.emit(r)

        def _done(res: Any) -> None:
            ok = bool(isinstance(res, dict) and res.get("ok"))
            if ok:
                try:
                    if btn_like is not None:
                        btn_like.setEnabled(False)
                    if btn_dislike is not None:
                        btn_dislike.setEnabled(False)
                except Exception:
                    pass
                try:
                    if int(score) == 1 and btn_like is not None:
                        btn_like.setStyleSheet("background: rgba(16,185,129,0.55); border: 1px solid rgba(16,185,129,0.65);")
                    if int(score) == -1 and btn_dislike is not None:
                        btn_dislike.setStyleSheet("background: rgba(239,68,68,0.45); border: 1px solid rgba(239,68,68,0.60);")
                except Exception:
                    pass
            else:
                try:
                    setattr(label, "_feedback_sent", False)
                except Exception:
                    pass
                try:
                    if btn_like is not None:
                        btn_like.setEnabled(True)
                    if btn_dislike is not None:
                        btn_dislike.setEnabled(True)
                except Exception:
                    pass
                try:
                    err = str(res.get("error") if isinstance(res, dict) else res)
                    label.setToolTip(f"feedback failed: {err}")
                except Exception:
                    pass

        t = _FbThread(self, mid, int(score), comment)
        t.finished.connect(_done)
        t.start()
        self._fb_thread = t

    def _post_chat(self, message: str) -> dict:
        url = f"{self._api_base}/api/v1/chat"
        payload = urllib.parse.quote("", safe="")
        del payload
        ctx = {
            "client": "desktop",
            "session_id": str(getattr(self, "_session_id", "")),
            "theme": str(getattr(self, "_theme", "dark")),
            "user_role": "Sensei (Teacher)",
            "current_mood": "Gentle",
        }
        data = json.dumps({"message": str(message), "context": ctx}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=180.0) as resp:  # 180s for 4-bit model loading
            raw = resp.read().decode("utf-8", errors="replace")
        obj: Any = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("reply"), str):
            return {
                "reply": str(obj.get("reply") or ""),
                "message_id": str(obj.get("message_id") or "").strip() or None,
            }
        return {"reply": "收到。", "message_id": None}

    def send_message(self) -> None:
        text = self.input_box.text().strip()
        if not text:
            return
        self.input_box.clear()

        self._append_chat(role="user", text=text)
        assistant_label = self._append_chat(role="assistant", text="", message_id=None, enable_feedback=True)

        # Hide the assistant bubble row until voice actually starts.
        _bubble = assistant_label.parentWidget()
        _row = _bubble.parentWidget() if _bubble is not None else None
        try:
            if _row is not None:
                _row.setVisible(False)
            elif _bubble is not None:
                _bubble.setVisible(False)
        except Exception:
            pass

        state: dict[str, bool] = {"started": False, "revealed": False}

        def _reveal(*, start_talking: bool) -> None:
            if state.get("revealed"):
                if start_talking and (not self.is_talking):
                    self.is_talking = True
                    self.mouth_timer.start(50)
                return
            state["revealed"] = True
            try:
                if _row is not None:
                    _row.setVisible(True)
                elif _bubble is not None:
                    _bubble.setVisible(True)
            except Exception:
                pass
            assistant_label.setText(self._pending_reply)
            safe = self._pending_reply.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
            self.avatar_view.page().runJavaScript(f"window.showBubble('{safe}');")
            if start_talking:
                self.is_talking = True
                self.mouth_timer.start(50)

        def _on_voice_started() -> None:
            state["started"] = True
            _reveal(start_talking=True)

        def _done(resp: Any) -> None:
            reply_text = ""
            message_id: str | None = None
            if isinstance(resp, dict):
                reply_text = str(resp.get("reply") or "")
                mid = str(resp.get("message_id") or "").strip()
                message_id = mid or None
            else:
                reply_text = str(resp)
            display_text = str(reply_text)
            cleaned, actions = self._extract_action_blocks(display_text)
            action_notes: list[str] = []
            if actions:
                for a in actions:
                    try:
                        act_name = str(a.get("action") or "")
                        is_ui = str(act_name).lower().startswith("ui.")
                        if is_ui:
                            res = self._execute_ui_action(a)
                            ok = bool(res.get("ok"))
                        else:
                            if not self._auto_execute_actions:
                                if self._show_action_notes:
                                    action_notes.append(f"[Action] {act_name} -> SKIP")
                                continue
                            res = self._post_action(a)
                            ok = bool(res.get("ok"))

                        # Only show notes when enabled, or when failed.
                        if self._show_action_notes or (not ok):
                            action_notes.append(f"[Action] {act_name} -> {'OK' if ok else 'FAIL'}")
                    except Exception as e:
                        if self._show_action_notes:
                            action_notes.append(f"[Action] {a.get('action')} -> ERROR: {e}")
            if action_notes:
                cleaned = (cleaned + "\n\n" + "\n".join(action_notes)).strip()

            self._pending_reply = cleaned

            # Enable feedback buttons when message_id is available.
            try:
                if message_id:
                    assistant_label.setProperty("message_id", message_id)
                    btn_like = getattr(assistant_label, "_btn_like", None)
                    btn_dislike = getattr(assistant_label, "_btn_dislike", None)
                    if btn_like is not None:
                        btn_like.setEnabled(True)
                    if btn_dislike is not None:
                        btn_dislike.setEnabled(True)
            except Exception:
                pass

            preset = self._pick_tts_preset(reply_text)
            self.tts_thread = TTSThread(
                cleaned,
                voice=str(getattr(self, "_tts_voice", "zh-CN-XiaoxiaoNeural")),
                backend=str(getattr(self, "_tts_backend", "edge")),
                gpt_sovits=dict(getattr(self, "_gpt_sovits_cfg", {}) or {}),
                preset=str(preset),
            )
            self.tts_thread.started_signal.connect(_on_voice_started)

            def _on_voice_finished(err: str) -> None:
                # If TTS failed before audio actually started, reveal the text as an error fallback.
                if (not state.get("started")) and (not state.get("revealed")):
                    _reveal(start_talking=False)
                self._on_tts_finished(err)

            self.tts_thread.finished_signal.connect(_on_voice_finished)
            self.tts_thread.start()

        class _ChatThread(QThread):
            finished = Signal(object)

            def __init__(self, outer: "MainWindow", msg: str):
                super().__init__()
                self.outer = outer
                self.msg = msg

            def run(self) -> None:
                try:
                    r = self.outer._post_chat(self.msg)
                except Exception as e:
                    r = {"reply": f"请求失败: {e}", "message_id": None}
                self.finished.emit(r)

        t = _ChatThread(self, text)
        t.finished.connect(_done)
        t.start()
        self._chat_thread = t

    def _update_mouth(self) -> None:
        if not self.is_talking:
            self.avatar_view.page().runJavaScript("window.setMouthOpen(0);")
            self.mouth_timer.stop()
            return

        val = random.uniform(0.0, 0.85)
        self.avatar_view.page().runJavaScript(f"window.setMouthOpen({val});")

    def _on_tts_finished(self, err: str) -> None:
        if err:
            print(err)
        self.is_talking = False
        self._update_mouth()

    @staticmethod
    def _http_ok(url: str, timeout_s: float = 1.0) -> bool:
        try:
            with urllib.request.urlopen(str(url), timeout=float(timeout_s)) as resp:
                return 200 <= int(getattr(resp, "status", 200)) < 400
        except (urllib.error.URLError, ValueError):
            return False

    @staticmethod
    def _wait_http(url: str, timeout_s: float = 30.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < float(timeout_s):
            if MainWindow._http_ok(url, timeout_s=1.0):
                return True
            time.sleep(0.3)
        return False

    def _start_services(self, *, api_host: str, api_port: int, ui_host: str, ui_port: int) -> None:
        api_url = f"http://{api_host}:{int(api_port)}/"
        ui_url = f"http://{ui_host}:{int(ui_port)}"

        # If already running, reuse.
        if not self._http_ok(api_url, timeout_s=0.5):
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "src.api.main:app",
                "--host",
                str(api_host),
                "--port",
                str(int(api_port)),
            ]
            env = dict(os.environ)
            env.setdefault("PYTHONUNBUFFERED", "1")
            self._api_proc = subprocess.Popen(cmd, cwd=str(self._repo_root()), env=env)
            if not self._wait_http(api_url, timeout_s=30.0):
                raise SystemExit(f"FastAPI did not start on {api_url}")

        if not self._http_ok(ui_url, timeout_s=0.5):
            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "src/ui/streamlit_app.py",
                "--server.address",
                str(ui_host),
                "--server.port",
                str(int(ui_port)),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ]
            env = dict(os.environ)
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
            env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
            self._ui_proc = subprocess.Popen(cmd, cwd=str(self._repo_root()), env=env)
            if not self._wait_http(ui_url, timeout_s=60.0):
                raise SystemExit(f"Streamlit did not start on {ui_url}")

    @staticmethod
    def _repo_root() -> str:
        # file = <repo>/src/ui/desktop/main.py
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    def _start_avatar_server(self, *, avatar_model: str | None) -> str:
        web_root = Path(__file__).resolve().parent / "web"

        model_path: Path | None = None
        if avatar_model:
            model_path = Path(str(avatar_model)).resolve()
            if not model_path.exists():
                model_path = None

        class _MuxHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(web_root), **kwargs)

            def log_message(self, fmt: str, *args) -> None:
                return

            def translate_path(self, path: str) -> str:
                u = urllib.parse.urlparse(path)
                req_path = urllib.parse.unquote(u.path or "/")

                if model_path is not None and req_path.startswith("/model/"):
                    rel = req_path[len("/model/") :].lstrip("/\\")
                    base = model_path.parent
                    cand = (base / rel).resolve()
                    if str(cand).startswith(str(base)):
                        return str(cand)
                    return str(base)

                return super().translate_path(path)

        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MuxHandler)
        self._web_server = httpd
        self._web_port = int(httpd.server_address[1])

        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._web_thread = t
        t.start()

        model_query = ""
        if model_path is not None:
            model_name = urllib.parse.quote(model_path.name)
            model_query = f"?model=/model/{model_name}"

        return f"http://127.0.0.1:{self._web_port}/index.html{model_query}"

    def closeEvent(self, event) -> None:  # type: ignore[override]
        for p in [self._ui_proc, self._api_proc]:
            if p is None:
                continue
            try:
                p.terminate()
            except Exception:
                pass

        if self._web_server is not None:
            with contextlib.suppress(Exception):
                self._web_server.shutdown()
            with contextlib.suppress(Exception):
                self._web_server.server_close()
            self._web_server = None
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-tower-url", default="http://127.0.0.1:8501")
    parser.add_argument("--start-services", action="store_true")
    parser.add_argument("--avatar-model", default="")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8501)
    args = parser.parse_args()

    preferred_model = Path(r"D:\Project\Stock\玛丽偶像 _vts\玛丽公开模型.model3.json")
    fallback_model = Path(r"D:\Project\Stock\长离带水印\长离带水印\长离.model3.json")

    if preferred_model.exists():
        avatar_model = str(preferred_model)
    elif fallback_model.exists():
        avatar_model = str(fallback_model)
    else:
        avatar_model = ""

    if str(args.avatar_model).strip():
        avatar_model = str(args.avatar_model).strip()

    app = QApplication(sys.argv)
    w = MainWindow(
        control_tower_url=str(args.control_tower_url),
        start_services=bool(args.start_services),
        avatar_model=str(avatar_model) if avatar_model else None,
        api_host=str(args.api_host),
        api_port=int(args.api_port),
        ui_host=str(args.ui_host),
        ui_port=int(args.ui_port),
    )
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
