import json
import subprocess
from pathlib import Path

from fact_checker.config.settings import get_settings


class MediaService:
    def __init__(self) -> None:
        settings = get_settings()
        self.ffmpeg_bin = settings.ffmpeg_bin
        self.ffprobe_bin = settings.ffprobe_bin

    def probe(self, input_path: str | Path) -> dict:
        command = [
            self.ffprobe_bin,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(input_path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    def extract_mono_wav(self, input_path: str | Path, output_path: str | Path) -> None:
        command = [
            self.ffmpeg_bin,
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ]
        subprocess.run(command, check=True)
