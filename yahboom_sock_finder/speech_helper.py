# speech_helper.py
import os
from playsound import playsound
from ament_index_python.packages import get_package_share_directory


class SpeechHelper:
    def __init__(self, node=None, language="en", package_name="largemodel"):
        self.node = node
        pkg_path = get_package_share_directory(package_name)

        self.audio_dict = {
            "longwan-women-1": os.path.join(pkg_path, "resources_file", "longwan-women-1.mp3"),
            "longwan-women-2": os.path.join(pkg_path, "resources_file", "longwan-women-2.mp3"),
            "longxiaochun-women-1": os.path.join(pkg_path, "resources_file", "longxiaochun-women-1.mp3"),
            "longxiaochun-women-2": os.path.join(pkg_path, "resources_file", "longxiaochun-women-2.mp3"),
        }

        if language == "zh":
            self.first_response = "longwan-women-1"
            self.error_response = "longwan-women-2"
        elif language == "en":
            self.first_response = "longxiaochun-women-1"
            self.error_response = "longxiaochun-women-2"
        else:
            raise ValueError("language must be 'en' or 'zh'")

    def _log(self, msg: str):
        if self.node is not None:
            self.node.get_logger().info(msg)
        else:
            print(msg)

    def play_first_response(self):
        path = self.audio_dict[self.first_response]
        self._log(f"[SpeechHelper] playing: {path}")
        playsound(path)

    def play_error_response(self):
        path = self.audio_dict[self.error_response]
        self._log(f"[SpeechHelper] playing: {path}")
        playsound(path)

    def play_key(self, key: str):
        path = self.audio_dict[key]
        self._log(f"[SpeechHelper] playing key={key}: {path}")
        playsound(path)

    def play_file(self, path: str):
        self._log(f"[SpeechHelper] playing custom file: {path}")
        playsound(path)