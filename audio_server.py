#!/usr/bin/env python3
"""
Kashi audio server — captions, audio, and lyric alignment via yt-dlp.

Setup (one time):
    pip install yt-dlp faster-whisper

Run:
    python audio_server.py

Endpoints:
  GET  /captions?v=ID    — download YouTube captions (yt-dlp) → JSON timestamps
  GET  /?v=ID            — stream audio for Whisper fallback
  POST /align            — download + transcribe + align → JSON timestamps
"""

import json, os, re, subprocess, sys, tempfile, unicodedata
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

PORT = 7755

# ── Optional faster-whisper ──────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    _whisper_model = None
    HAS_FASTER_WHISPER = True
    print("faster-whisper available — Whisper alignment enabled.")
except ImportError:
    HAS_FASTER_WHISPER = False
    print("faster-whisper not found (pip install faster-whisper for Whisper fallback).")


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("  Loading Whisper model (~300 MB, first run only)…")
        _whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
        print("  Model ready.")
    return _whisper_model


# ── Caption fetching via yt-dlp ──────────────────────────────────────────────

def fetch_captions(video_id):
    """
    Download Japanese captions for a YouTube video using yt-dlp.
    Returns (lines, source) where lines = [{text, startMs, endMs}]
    and source is 'manual' | 'asr' | None.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "%(id)s.%(ext)s")
        subprocess.run(
            ["yt-dlp",
             "--write-sub", "--write-auto-sub",
             "--sub-lang", "ja",
             "--sub-format", "json3",
             "--skip-download", "--no-playlist",
             "-o", out, url],
            capture_output=True, timeout=45, check=False,
        )
        files = os.listdir(tmpdir)
        # Manual captions first, then auto-generated
        ordered = (
            [f for f in files if f.endswith(".json3") and ".auto." not in f.lower()] +
            [f for f in files if f.endswith(".json3") and ".auto." in f.lower()]
        )
        for fname in ordered:
            is_auto = ".auto." in fname.lower()
            try:
                with open(os.path.join(tmpdir, fname), encoding="utf-8") as fp:
                    data = json.load(fp)
                lines = parse_json3(data)
                if lines:
                    print(f"    yt-dlp captions: {len(lines)} lines ({'asr' if is_auto else 'manual'})")
                    return lines, "asr" if is_auto else "manual"
            except Exception as e:
                print(f"    parse error {fname}: {e}")
    return None, None


def parse_json3(data):
    """Parse yt-dlp's json3 caption format into line dicts."""
    lines = []
    for ev in data.get("events", []):
        segs = ev.get("segs", [])
        text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        if not text:
            continue
        start = int(ev.get("tStartMs", 0))
        dur   = int(ev.get("dDurationMs", 3000))
        lines.append({"text": text, "startMs": start, "endMs": start + dur})
    return lines or None


# ── Audio download ────────────────────────────────────────────────────────────

def download_audio(video_id, out_path):
    subprocess.run(
        ["yt-dlp", "-f", "bestaudio/best", "--extract-audio",
         "--audio-format", "mp3", "--audio-quality", "5",
         "--no-playlist", "-o", out_path,
         f"https://www.youtube.com/watch?v={video_id}"],
        check=True, capture_output=True,
    )


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe_local(audio_path, lyrics_lines):
    model  = get_whisper_model()
    prompt = "\n".join(lyrics_lines[:30])[:500]
    segs, _ = model.transcribe(
        audio_path, language="ja", word_timestamps=True,
        initial_prompt=prompt, beam_size=5, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    words = []
    for seg in segs:
        for w in (seg.words or []):
            words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
    return words


def transcribe_groq(audio_path, lyrics_lines, groq_key):
    import urllib.request
    prompt   = "\n".join(lyrics_lines[:8])[:224]
    boundary = "----KashiBoundary"
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    def field(name, value):
        return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode()

    body = (
        field("model",  "whisper-large-v3") +
        field("language", "ja") +
        field("response_format", "verbose_json") +
        field("timestamp_granularities[]", "word") +
        field("timestamp_granularities[]", "segment") +
        (field("prompt", prompt) if prompt else b"") +
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.mp3\"\r\nContent-Type: audio/mpeg\r\n\r\n").encode() +
        audio_data +
        f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/audio/transcriptions", data=body,
        headers={"Authorization": f"Bearer {groq_key}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read()).get("words") or []


# ── Alignment ─────────────────────────────────────────────────────────────────

def normalize(text):
    text = unicodedata.normalize("NFKC", text or "")
    out  = []
    for c in text:
        cp = ord(c)
        out.append(chr(cp - 0x60) if 0x30A1 <= cp <= 0x30F6 else c)
    return re.sub(r"[^぀-ゟ一-鿿㐀-䶿a-z0-9]", "", "".join(out).lower())


def align(lyrics_lines, words):
    if not words or not lyrics_lines:
        n = len(lyrics_lines)
        return [{"startMs": i * 4000, "endMs": (i + 1) * 4000} for i in range(n)]

    norm_words = [{"norm": normalize(w.get("word", "")), "start": w["start"]} for w in words]
    norm_words = [w for w in norm_words if w["norm"]]
    full       = "".join(w["norm"] for w in norm_words)
    pos2word   = []
    for wi, w in enumerate(norm_words):
        pos2word.extend([wi] * len(w["norm"]))

    anchors = []
    cursor  = 0
    for li, line in enumerate(lyrics_lines):
        norm = normalize(line)
        if not norm:
            continue
        for plen in range(min(len(norm), 12), 1, -1):
            idx = full.find(norm[:plen], cursor)
            if idx < 0:
                continue
            wi = pos2word[idx] if idx < len(pos2word) else 0
            anchors.append((li, round(norm_words[wi]["start"] * 1000)))
            cursor = idx + 1
            break

    if not anchors:
        total = round(words[-1]["end"] * 1000) if words else len(lyrics_lines) * 4000
        n     = len(lyrics_lines)
        return [{"startMs": round(i * total / n), "endMs": round((i + 1) * total / n)} for i in range(n)]

    n     = len(lyrics_lines)
    raw   = [None] * n
    for li, ms in anchors:
        raw[li] = ms

    avg_ms = 3000
    if len(anchors) > 1:
        span_li = anchors[-1][0] - anchors[0][0]
        span_ms = anchors[-1][1] - anchors[0][1]
        if span_li > 0:
            avg_ms = span_ms / span_li

    first_li, first_ms = anchors[0]
    for i in range(first_li):
        raw[i] = round((i / first_li) * first_ms) if first_li > 0 else 0

    for k in range(len(anchors) - 1):
        a_li, a_ms = anchors[k]
        b_li, b_ms = anchors[k + 1]
        for i in range(a_li + 1, b_li):
            raw[i] = round(a_ms + ((i - a_li) / (b_li - a_li)) * (b_ms - a_ms))

    last_li, last_ms = anchors[-1]
    for i in range(last_li + 1, n):
        raw[i] = round(last_ms + (i - last_li) * avg_ms)

    for i in range(1, n):
        if raw[i] is None or raw[i] <= raw[i - 1]:
            raw[i] = raw[i - 1] + 300

    return [{"startMs": raw[i], "endMs": raw[i + 1] if i + 1 < n else raw[i] + 8000}
            for i in range(n)]


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} — {fmt % args}")

    def cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200); self.cors(); self.end_headers()

    # ── GET /captions?v=ID  →  caption lines with timestamps ─────────────────
    # ── GET /?v=ID           →  stream audio ─────────────────────────────────
    def do_GET(self):
        parsed   = urlparse(self.path)
        params   = parse_qs(parsed.query)
        video_id = params.get("v", [""])[0]

        if parsed.path == "/captions":
            if not video_id:
                return self._json(400, {"error": "Missing ?v=VIDEO_ID"})
            print(f"  /captions {video_id}")
            try:
                lines, source = fetch_captions(video_id)
                self._json(200, {"lines": lines or [], "source": source or "none"})
            except Exception as e:
                print(f"  caption error: {e}")
                self._json(500, {"error": str(e), "lines": [], "source": "error"})
            return

        # Audio stream
        if not video_id:
            return self._err(400, "Missing ?v=VIDEO_ID")
        print(f"  Streaming audio for {video_id}…")
        try:
            proc = subprocess.Popen(
                ["yt-dlp", "-f", "bestaudio/best", "--extract-audio",
                 "--audio-format", "mp3", "--audio-quality", "5",
                 "--no-playlist", "-o", "-",
                 f"https://www.youtube.com/watch?v={video_id}"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.send_response(200); self.cors()
            self.send_header("Content-Type", "audio/mpeg")
            self.end_headers()
            while chunk := proc.stdout.read(65536):
                self.wfile.write(chunk)
            proc.wait()
        except BrokenPipeError:
            pass
        except Exception as e:
            print(f"  audio error: {e}")

    # ── POST /align  →  JSON timestamps ──────────────────────────────────────
    def do_POST(self):
        if urlparse(self.path).path != "/align":
            return self._err(404, "Not found")
        length   = int(self.headers.get("Content-Length", 0))
        body     = json.loads(self.rfile.read(length))
        video_id = body.get("videoId", "")
        lyrics   = body.get("lyrics", [])
        groq_key = body.get("groqKey", "")
        if not video_id or not lyrics:
            return self._err(400, "videoId and lyrics required")

        print(f"  /align {video_id} ({len(lyrics)} lines)")
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                audio_path = f.name
            try:
                print("    Downloading audio…")
                download_audio(video_id, audio_path)
                if HAS_FASTER_WHISPER:
                    print("    Transcribing (faster-whisper)…")
                    whisper_words = transcribe_local(audio_path, lyrics)
                elif groq_key:
                    print("    Transcribing (Groq)…")
                    whisper_words = transcribe_groq(audio_path, lyrics, groq_key)
                else:
                    return self._err(400, "No groqKey and faster-whisper not installed")
                print(f"    {len(whisper_words)} words → aligning…")
                timestamps = align(lyrics, whisper_words)
            finally:
                try: os.unlink(audio_path)
                except OSError: pass

            payload = json.dumps({"timestamps": timestamps}).encode()
            self.send_response(200); self.cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            print(f"    Done.")

        except subprocess.CalledProcessError as e:
            msg = (e.stderr or b"").decode(errors="replace")
            print(f"  yt-dlp error: {msg[:200]}")
            self._err(500, f"yt-dlp failed: {msg[:200]}")
        except Exception as e:
            import traceback; traceback.print_exc()
            self._err(500, str(e))

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code); self.cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json(code, {"error": msg})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("yt-dlp not found.  Install:  pip install yt-dlp"); sys.exit(1)

    print(f"\nKashi audio server  http://localhost:{PORT}")
    print("  /captions  — yt-dlp caption download (primary timing source)")
    print("  /align     — Whisper forced alignment (fallback)")
    if HAS_FASTER_WHISPER:
        print("  Transcription: local faster-whisper")
    else:
        print("  Transcription: Groq API  (pip install faster-whisper for local)")
    print("\nPress Ctrl+C to stop.\n")

    try:
        HTTPServer(("localhost", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
