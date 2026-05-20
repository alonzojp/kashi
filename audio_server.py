#!/usr/bin/env python3
"""
Kashi audio server — proper lyric timing via vocal separation + forced alignment.

Setup (one time):
    pip install yt-dlp faster-whisper demucs

demucs separates vocals from instruments so Whisper gets a clean vocal track.
Without it alignment is unreliable. Install it — it makes a massive difference.

Run:
    python audio_server.py

Endpoints:
  GET  /captions?v=ID    — yt-dlp caption download (fast, when available)
  GET  /?v=ID            — stream raw audio
  POST /align            — vocal separation → Whisper → forced alignment → timestamps
"""

import json, os, re, subprocess, sys, tempfile, unicodedata
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

PORT = 7755

# ── Capability detection ─────────────────────────────────────────────────────

def _check(cmd):
    try:
        subprocess.run(cmd, capture_output=True, timeout=5, check=True)
        return True
    except Exception:
        return False

HAS_DEMUCS        = _check(['python', '-m', 'demucs', '--help'])
HAS_FASTER_WHISPER = False
_whisper_model     = None

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    pass

print("\n── Kashi audio server ──────────────────────────────────────────")
print(f"  Vocal separation (demucs):  {'✓ installed' if HAS_DEMUCS else '✗ missing  ← pip install demucs'}")
print(f"  Transcription (faster-whisper): {'✓ installed' if HAS_FASTER_WHISPER else '✗ missing  ← pip install faster-whisper'}")
if not HAS_DEMUCS:
    print("\n  ⚠  demucs is strongly recommended — without it Whisper")
    print("     gets confused by instruments and timing will be off.")
print("────────────────────────────────────────────────────────────────\n")


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("  Loading Whisper model (downloads ~300 MB on first run)…")
        _whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
        print("  Whisper ready.")
    return _whisper_model


# ── yt-dlp helpers ───────────────────────────────────────────────────────────

def download_audio(video_id, out_path):
    subprocess.run(
        ["yt-dlp", "-f", "bestaudio/best", "--extract-audio",
         "--audio-format", "wav",          # WAV so demucs gets clean input
         "--no-playlist", "-o", out_path,
         f"https://www.youtube.com/watch?v={video_id}"],
        check=True, capture_output=True,
    )


def fetch_captions(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "%(id)s.%(ext)s")
        subprocess.run(
            ["yt-dlp", "--write-sub", "--write-auto-sub",
             "--sub-lang", "ja", "--sub-format", "json3",
             "--skip-download", "--no-playlist", "-o", out, url],
            capture_output=True, timeout=45, check=False,
        )
        files = os.listdir(tmpdir)
        ordered = (
            [f for f in files if f.endswith(".json3") and ".auto." not in f.lower()] +
            [f for f in files if f.endswith(".json3") and ".auto."     in f.lower()]
        )
        for fname in ordered:
            is_auto = ".auto." in fname.lower()
            try:
                with open(os.path.join(tmpdir, fname), encoding="utf-8") as fp:
                    data = json.load(fp)
                lines = _parse_json3(data)
                if lines:
                    print(f"    yt-dlp captions: {len(lines)} lines ({'asr' if is_auto else 'manual'})")
                    return lines, "asr" if is_auto else "manual"
            except Exception as e:
                print(f"    caption parse error: {e}")
    return None, None


def _parse_json3(data):
    lines = []
    for ev in data.get("events", []):
        text = "".join(s.get("utf8", "") for s in ev.get("segs", [])).replace("\n", " ").strip()
        if not text:
            continue
        start = int(ev.get("tStartMs", 0))
        dur   = int(ev.get("dDurationMs", 3000))
        lines.append({"text": text, "startMs": start, "endMs": start + dur})
    return lines or None


# ── Vocal separation ─────────────────────────────────────────────────────────

def separate_vocals(audio_path, work_dir):
    """
    Run demucs htdemucs to extract the vocal stem.
    Returns path to vocals.wav, or None on failure.
    """
    print("  Separating vocals (demucs)… this takes 2–5 min on CPU")
    try:
        subprocess.run(
            ["python", "-m", "demucs",
             "--two-stems", "vocals",   # only produce vocals + no_vocals
             "--out", work_dir,
             audio_path],
            check=True, capture_output=True, timeout=900,
        )
        # demucs outputs to: work_dir/htdemucs/<filename_without_ext>/vocals.wav
        for root, _, files in os.walk(work_dir):
            for f in files:
                if f == "vocals.wav":
                    path = os.path.join(root, f)
                    size = os.path.getsize(path)
                    print(f"  Vocals extracted ({size // 1024} KB): {path}")
                    return path
        print("  demucs ran but vocals.wav not found")
    except subprocess.TimeoutExpired:
        print("  demucs timed out (> 15 min) — falling back to full mix")
    except subprocess.CalledProcessError as e:
        print(f"  demucs error: {e.stderr.decode(errors='replace')[:300]}")
    except Exception as e:
        print(f"  demucs failed: {e}")
    return None


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe_local(audio_path, lyrics_hint):
    """
    Transcribe with faster-whisper. lyrics_hint biases the model toward
    the expected text (improves accuracy for music).
    Returns [{word, start, end}].
    """
    model  = get_whisper_model()
    # Use first 30 lyric lines as prompt so Whisper learns the vocabulary
    prompt = "\n".join(lyrics_hint[:30])[:500]
    print(f"  Transcribing with faster-whisper (prompt: {len(prompt)} chars)…")

    segs, _ = model.transcribe(
        audio_path,
        language        = "ja",
        word_timestamps = True,
        initial_prompt  = prompt,
        beam_size       = 5,
        vad_filter      = True,           # skip silent / non-vocal sections
        vad_parameters  = {"min_silence_duration_ms": 300},
    )
    words = []
    for seg in segs:
        for w in (seg.words or []):
            if w.word.strip():
                words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
    print(f"  Got {len(words)} words from Whisper")
    return words


def transcribe_groq(audio_path, lyrics_hint, groq_key):
    """Fallback: Groq Whisper API."""
    import urllib.request
    prompt   = "\n".join(lyrics_hint[:8])[:224]
    boundary = "KashiBoundary7755"
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    def field(name, val):
        return f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n".encode()

    body = (
        field("model", "whisper-large-v3") +
        field("language", "ja") +
        field("response_format", "verbose_json") +
        field("timestamp_granularities[]", "word") +
        field("timestamp_granularities[]", "segment") +
        (field("prompt", prompt) if prompt else b"") +
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode() +
        audio_data +
        f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/audio/transcriptions", data=body,
        headers={"Authorization": f"Bearer {groq_key}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    print("  Transcribing via Groq API…")
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read())
    words = result.get("words") or []
    print(f"  Got {len(words)} words from Groq")
    return words


# ── Alignment ─────────────────────────────────────────────────────────────────

def normalize(text):
    """Normalize Japanese text: katakana→hiragana, fullwidth→ascii, strip punctuation."""
    text = unicodedata.normalize("NFKC", text or "")
    out  = []
    for c in text:
        cp = ord(c)
        out.append(chr(cp - 0x60) if 0x30A1 <= cp <= 0x30F6 else c)
    return re.sub(r"[^぀-ゟ一-鿿㐀-䶿a-z0-9]", "", "".join(out).lower())


def align(lyrics_lines, words):
    """
    Align each lyric line to a Whisper word timestamp.

    Strategy:
    1. Build a normalized transcript string from all Whisper words.
    2. For each lyric line, search for the longest matching prefix in the
       transcript (trying lengths 15→2 chars), always advancing the cursor.
    3. Collect matched lines as anchor points with real timestamps.
    4. Interpolate/extrapolate for unmatched lines between anchors.
    5. Enforce strict monotonicity with a minimum 200ms gap.

    With clean vocal audio from demucs, Whisper's transcript is close
    enough to the lyrics that step 2 finds anchors for most lines.
    """
    if not words or not lyrics_lines:
        n = len(lyrics_lines)
        return [{"startMs": i * 4000, "endMs": (i + 1) * 4000} for i in range(n)]

    # Build normalized transcript with position→word-index map
    nwords    = [{"n": normalize(w.get("word", "")), "start": w["start"]} for w in words]
    nwords    = [w for w in nwords if w["n"]]
    full      = "".join(w["n"] for w in nwords)
    pos2wi    = []
    for wi, w in enumerate(nwords):
        pos2wi.extend([wi] * len(w["n"]))

    # ── Anchor pass ───────────────────────────────────────────────────────────
    anchors = []   # [(line_idx, start_ms)]
    cursor  = 0

    for li, line in enumerate(lyrics_lines):
        norm = normalize(line)
        if not norm:
            continue
        # Try prefix lengths longest→shortest
        for plen in range(min(len(norm), 15), 1, -1):
            idx = full.find(norm[:plen], cursor)
            if idx < 0:
                continue
            wi = pos2wi[idx] if idx < len(pos2wi) else 0
            start_ms = round(nwords[wi]["start"] * 1000)
            # Sanity check: timestamp must be >= last anchor's timestamp
            if anchors and start_ms < anchors[-1][1]:
                continue
            anchors.append((li, start_ms))
            cursor = idx + 1
            break

    anchor_rate = len(anchors) / max(len(lyrics_lines), 1)
    print(f"  Alignment: {len(anchors)}/{len(lyrics_lines)} anchors ({anchor_rate:.0%})")

    if not anchors:
        # No anchors at all — distribute evenly across audio duration
        total = round(words[-1].get("end", words[-1]["start"] + 1) * 1000)
        n     = len(lyrics_lines)
        return [{"startMs": round(i * total / n), "endMs": round((i + 1) * total / n)}
                for i in range(n)]

    # ── Interpolation ─────────────────────────────────────────────────────────
    n     = len(lyrics_lines)
    raw   = [None] * n
    for li, ms in anchors:
        raw[li] = ms

    # Average ms per line (for extrapolation)
    avg_ms = 3000
    if len(anchors) > 1:
        span_li = anchors[-1][0] - anchors[0][0]
        span_ms = anchors[-1][1] - anchors[0][1]
        if span_li > 0:
            avg_ms = span_ms / span_li

    # Before first anchor: spread proportionally from 0
    first_li, first_ms = anchors[0]
    for i in range(first_li):
        raw[i] = round((i / first_li) * first_ms) if first_li > 0 else 0

    # Between consecutive anchors: linear interpolation
    for k in range(len(anchors) - 1):
        a_li, a_ms = anchors[k]
        b_li, b_ms = anchors[k + 1]
        for i in range(a_li + 1, b_li):
            t      = (i - a_li) / (b_li - a_li)
            raw[i] = round(a_ms + t * (b_ms - a_ms))

    # After last anchor: linear extrapolation
    last_li, last_ms = anchors[-1]
    for i in range(last_li + 1, n):
        raw[i] = round(last_ms + (i - last_li) * avg_ms)

    # Fill any remaining None and enforce strict monotonicity (min 200ms gap)
    for i in range(n):
        if raw[i] is None:
            raw[i] = (raw[i - 1] + 200) if i > 0 else 0
    for i in range(1, n):
        if raw[i] <= raw[i - 1]:
            raw[i] = raw[i - 1] + 200

    result = []
    for i, ms in enumerate(raw):
        end = raw[i + 1] if i + 1 < n else ms + 8000
        result.append({"startMs": ms, "endMs": end})
    return result


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

    def do_GET(self):
        parsed   = urlparse(self.path)
        params   = parse_qs(parsed.query)
        video_id = params.get("v", [""])[0]

        # /captions — download YouTube captions via yt-dlp
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

        # / — stream audio
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
            print(f"  stream error: {e}")

    def do_POST(self):
        if urlparse(self.path).path != "/align":
            return self._err(404, "Not found")
        body     = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        video_id = body.get("videoId", "")
        lyrics   = body.get("lyrics", [])
        groq_key = body.get("groqKey", "")

        if not video_id or not lyrics:
            return self._err(400, "videoId and lyrics required")

        print(f"\n  /align — {video_id}  ({len(lyrics)} lyric lines)")
        print(f"  Pipeline: yt-dlp → {'demucs → ' if HAS_DEMUCS else ''}faster-whisper → alignment")

        try:
            with tempfile.TemporaryDirectory() as work_dir:
                # ── 1. Download audio as WAV ──────────────────────────────
                raw_path = os.path.join(work_dir, "audio.wav")
                print("  Downloading audio…")
                download_audio(video_id, raw_path)
                size = os.path.getsize(raw_path) // 1024
                print(f"  Downloaded {size} KB")

                # ── 2. Separate vocals ────────────────────────────────────
                if HAS_DEMUCS:
                    vocals_path = separate_vocals(raw_path, work_dir)
                    transcribe_path = vocals_path or raw_path
                    if not vocals_path:
                        print("  Falling back to full mix (vocals separation failed)")
                else:
                    transcribe_path = raw_path
                    print("  ⚠  demucs not installed — transcribing full mix (less accurate)")
                    print("     Install it for much better results:  pip install demucs")

                # ── 3. Transcribe ─────────────────────────────────────────
                if HAS_FASTER_WHISPER:
                    words = transcribe_local(transcribe_path, lyrics)
                elif groq_key:
                    words = transcribe_groq(transcribe_path, lyrics, groq_key)
                else:
                    return self._err(400, "faster-whisper not installed and no groqKey provided")

                if not words:
                    return self._err(500, "Whisper returned no words")

                # ── 4. Align ──────────────────────────────────────────────
                timestamps = align(lyrics, words)

            payload = json.dumps({"timestamps": timestamps}).encode()
            self.send_response(200); self.cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            print(f"  Done — {len(timestamps)} lines aligned.\n")

        except subprocess.CalledProcessError as e:
            msg = (e.stderr or b"").decode(errors="replace")[:300]
            print(f"  yt-dlp error: {msg}")
            self._err(500, f"Download failed: {msg}")
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
        print("yt-dlp not found.  pip install yt-dlp"); sys.exit(1)

    if not HAS_FASTER_WHISPER:
        print("faster-whisper not found.  pip install faster-whisper")
        print("(needed for transcription — without it only Groq API fallback works)\n")

    print(f"Listening on http://localhost:{PORT}")
    print("Press Ctrl+C to stop.\n")
    try:
        HTTPServer(("localhost", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
