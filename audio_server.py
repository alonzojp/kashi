#!/usr/bin/env python3
"""
Kashi audio server — streams YouTube audio to the browser for Whisper sync.
Requires yt-dlp:  pip install yt-dlp

Run:  python audio_server.py
Then load a YouTube video in Kashi and auto-sync will work automatically.
"""

import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

PORT = 7755


def check_ytdlp():
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("yt-dlp not found. Install it with:  pip install yt-dlp")
        sys.exit(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} – {fmt % args}")

    def send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        video_id = params.get("v", [""])[0]

        if not video_id:
            self.send_response(400)
            self.send_cors()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Missing ?v=VIDEO_ID")
            return

        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"  Fetching audio for {video_id}…")

        try:
            proc = subprocess.Popen(
                [
                    "yt-dlp",
                    "-f", "bestaudio/best",
                    "--extract-audio",
                    "--audio-format", "mp3",
                    "--audio-quality", "5",   # ~128 kbps — enough for Whisper
                    "--no-playlist",
                    "-o", "-",
                    url,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.send_response(200)
            self.send_cors()
            self.send_header("Content-Type", "audio/mpeg")
            self.end_headers()

            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)

            proc.wait()

        except BrokenPipeError:
            pass  # client closed connection early
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    check_ytdlp()
    print(f"Kashi audio server running on http://localhost:{PORT}")
    print("Keep this window open while using Kashi. Press Ctrl+C to stop.\n")
    try:
        HTTPServer(("localhost", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
