#!/usr/bin/env python3
"""
Kashi audio server — downloads YouTube audio and aligns lyrics with accurate timestamps.

Setup (one time):
    pip install yt-dlp faster-whisper

Run:
    python audio_server.py

Keep this window open while using Kashi.

Endpoints:
  GET  /?v=VIDEO_ID          — streams audio (fallback, rarely used directly)
  POST /align                — full pipeline: download → transcribe → align → LRC timestamps
"""

import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

PORT = 7755

# ---------------------------------------------------------------------------
# Optional faster-whisper (local, no rate limits, uses lyrics as prompt)
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel
    _whisper_model = None          # loaded lazily on first /align request
    HAS_FASTER_WHISPER = True
    print("faster-whisper found — will use local transcription.")
except ImportError:
    HAS_FASTER_WHISPER = False
    print("faster-whisper not found — will use Groq API for transcription.")
    print("  For better accuracy:  pip install faster-whisper")


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("  Loading Whisper model (first run — downloads ~300 MB)…")
        _whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
        print("  Model ready.")
    return _whisper_model


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------

def download_audio(video_id: str, out_path: str):
    """Download best audio to out_path using yt-dlp."""
    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestaudio/best",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "5",
            "--no-playlist",
            "-o", out_path,
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_local(audio_path: str, lyrics_lines: list) -> list:
    """
    Transcribe with faster-whisper using lyrics as initial prompt.
    Returns list of {word, start, end} dicts.
    """
    model = get_whisper_model()
    prompt = "\n".join(lyrics_lines[:30])[:500]   # guide model with known text

    segments, _ = model.transcribe(
        audio_path,
        language="ja",
        word_timestamps=True,
        initial_prompt=prompt,
        beam_size=5,
        vad_filter=True,          # skip silent / non-speech sections
        vad_parameters={"min_silence_duration_ms": 500},
    )

    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
    return words


def transcribe_groq(audio_path: str, lyrics_lines: list, groq_key: str) -> list:
    """Fallback: Groq Whisper API — returns list of {word, start, end}."""
    import urllib.request

    prompt = "\n".join(lyrics_lines[:8])[:224]

    boundary = "----KashiBoundary"
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    def field(name, value):
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    body = (
        field("model", "whisper-large-v3")
        + field("language", "ja")
        + field("response_format", "verbose_json")
        + field("timestamp_granularities[]", "word")
        + field("timestamp_granularities[]", "segment")
        + (field("prompt", prompt) if prompt else b"")
        + (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="audio.mp3"\r\n'
            f"Content-Type: audio/mpeg\r\n\r\n"
        ).encode()
        + audio_data
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        data=body,
        headers={
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read())

    return result.get("words") or []


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalize Japanese text to hiragana + ASCII for comparison."""
    text = unicodedata.normalize("NFKC", text or "")
    out = []
    for c in text:
        cp = ord(c)
        if 0x30A1 <= cp <= 0x30F6:          # katakana → hiragana
            out.append(chr(cp - 0x60))
        else:
            out.append(c)
    text = "".join(out).lower()
    text = re.sub(r"[^぀-ゟ一-鿿㐀-䶿a-z0-9]", "", text)
    return text


def align(lyrics_lines: list, words: list) -> list:
    """
    Map each lyric line to a timestamp from the Whisper word list.

    Strategy:
      1. Build a normalized concatenation of Whisper words with position→word index map.
      2. For each lyric line try prefix matches (longest→shortest) against the
         normalized transcript, always advancing the search cursor forward.
      3. Collect anchor points (line_idx, start_ms).
      4. Interpolate or extrapolate for unmatched lines so every line has a
         unique, monotonically increasing timestamp.
    """
    if not words or not lyrics_lines:
        n = len(lyrics_lines)
        return [{"startMs": i * 4000, "endMs": (i + 1) * 4000} for i in range(n)]

    # Build normalized transcript string
    norm_words = [{"norm": normalize(w.get("word", "")), "start": w["start"]} for w in words]
    norm_words = [w for w in norm_words if w["norm"]]

    full = "".join(w["norm"] for w in norm_words)
    pos_to_word = []                       # char position → word index
    for wi, w in enumerate(norm_words):
        pos_to_word.extend([wi] * len(w["norm"]))

    # ── Anchor pass ──────────────────────────────────────────────────────────
    anchors = []   # list of (line_idx, start_ms)
    cursor = 0

    for li, line in enumerate(lyrics_lines):
        norm = normalize(line)
        if not norm:
            continue
        found = False
        for plen in range(min(len(norm), 12), 1, -1):
            idx = full.find(norm[:plen], cursor)
            if idx < 0:
                continue
            wi = pos_to_word[idx] if idx < len(pos_to_word) else 0
            start_ms = round(norm_words[wi]["start"] * 1000)
            anchors.append((li, start_ms))
            cursor = idx + 1
            found = True
            break

    if not anchors:
        # No anchors at all — distribute evenly across the audio duration
        total_ms = round(words[-1]["end"] * 1000) if words else len(lyrics_lines) * 4000
        n = len(lyrics_lines)
        return [
            {"startMs": round(i * total_ms / n), "endMs": round((i + 1) * total_ms / n)}
            for i in range(n)
        ]

    # ── Assign raw timestamps ─────────────────────────────────────────────────
    n = len(lyrics_lines)
    raw = [None] * n
    for li, ms in anchors:
        raw[li] = ms

    avg_ms = 3000
    if len(anchors) > 1:
        span_li = anchors[-1][0] - anchors[0][0]
        span_ms = anchors[-1][1] - anchors[0][1]
        if span_li > 0:
            avg_ms = span_ms / span_li

    # Before first anchor: proportional from 0 → first anchor time
    first_li, first_ms = anchors[0]
    for i in range(first_li):
        raw[i] = round((i / first_li) * first_ms) if first_li > 0 else 0

    # Between anchors: linear interpolation
    for k in range(len(anchors) - 1):
        a_li, a_ms = anchors[k]
        b_li, b_ms = anchors[k + 1]
        for i in range(a_li + 1, b_li):
            t = (i - a_li) / (b_li - a_li)
            raw[i] = round(a_ms + t * (b_ms - a_ms))

    # After last anchor: linear extrapolation
    last_li, last_ms = anchors[-1]
    for i in range(last_li + 1, n):
        raw[i] = round(last_ms + (i - last_li) * avg_ms)

    # Enforce strict monotonicity with minimum 300 ms gap
    for i in range(1, n):
        if raw[i] is None or raw[i] <= raw[i - 1]:
            raw[i] = raw[i - 1] + 300

    # Build result
    result = []
    for i, ms in enumerate(raw):
        end = raw[i + 1] if i + 1 < n else ms + 8000
        result.append({"startMs": ms, "endMs": end})
    return result


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} — {fmt % args}")

    def cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.cors()
        self.end_headers()

    # ── GET /?v=ID  →  stream audio ──────────────────────────────────────────
    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        video_id = params.get("v", [""])[0]
        if not video_id:
            self._err(400, "Missing ?v=VIDEO_ID")
            return

        print(f"  Streaming audio for {video_id}…")
        try:
            proc = subprocess.Popen(
                ["yt-dlp", "-f", "bestaudio/best", "--extract-audio",
                 "--audio-format", "mp3", "--audio-quality", "5",
                 "--no-playlist", "-o", "-",
                 f"https://www.youtube.com/watch?v={video_id}"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.send_response(200)
            self.cors()
            self.send_header("Content-Type", "audio/mpeg")
            self.end_headers()
            while chunk := proc.stdout.read(65536):
                self.wfile.write(chunk)
            proc.wait()
        except BrokenPipeError:
            pass
        except Exception as e:
            print(f"  Audio error: {e}")

    # ── POST /align  →  JSON timestamps ──────────────────────────────────────
    def do_POST(self):
        if urlparse(self.path).path != "/align":
            self._err(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        video_id  = body.get("videoId", "")
        lyrics    = body.get("lyrics", [])     # list of lyric line strings
        groq_key  = body.get("groqKey", "")

        if not video_id or not lyrics:
            self._err(400, "videoId and lyrics required")
            return

        print(f"  Aligning {len(lyrics)} lines for {video_id}…")

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                audio_path = f.name

            try:
                # 1. Download audio
                print("    Downloading audio…")
                download_audio(video_id, audio_path)

                # 2. Transcribe
                if HAS_FASTER_WHISPER:
                    print("    Transcribing (local faster-whisper)…")
                    whisper_words = transcribe_local(audio_path, lyrics)
                elif groq_key:
                    print("    Transcribing (Groq API)…")
                    whisper_words = transcribe_groq(audio_path, lyrics, groq_key)
                else:
                    self._err(400, "No groqKey provided and faster-whisper not installed")
                    return

                print(f"    Got {len(whisper_words)} words from Whisper.")

                # 3. Align
                print("    Aligning…")
                timestamps = align(lyrics, whisper_words)

            finally:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

            # 4. Return JSON
            payload = json.dumps({"timestamps": timestamps}).encode()
            self.send_response(200)
            self.cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            print(f"    Done — {len(timestamps)} lines aligned.")

        except subprocess.CalledProcessError as e:
            msg = e.stderr.decode(errors="replace") if e.stderr else str(e)
            print(f"  yt-dlp error: {msg}")
            self._err(500, f"yt-dlp failed: {msg[:200]}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._err(500, str(e))

    def _err(self, code, msg):
        body = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("yt-dlp not found.  Install:  pip install yt-dlp")
        sys.exit(1)

    print(f"\nKashi audio server on http://localhost:{PORT}")
    if HAS_FASTER_WHISPER:
        print("Mode: local faster-whisper (accurate, offline)")
    else:
        print("Mode: Groq API (requires key in Kashi settings)")
        print("  Upgrade:  pip install faster-whisper")
    print("Press Ctrl+C to stop.\n")

    try:
        HTTPServer(("localhost", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
