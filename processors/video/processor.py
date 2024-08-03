import os
from typing import TYPE_CHECKING
import tempfile

from loguru import logger
import yt_dlp
from moviepy.editor import *
import speech_recognition as sr

if TYPE_CHECKING:
    from processors.config import Config


class VideoProcessor:
    def __init__(self, config: "Config"):
        self.config = config
        self.timestamps = {}

    def get_timestamps(self, image_path: str) -> dict:
        image_name = os.path.split(image_path)[-1]
        return self.timestamps.get(image_name, {})

    @staticmethod
    def download_video(url, output_path):
        """
        Download a video from a given url and save it to the output path.

        Parameters:
        url (str): The url of the video to download.
        output_path (str): The path to save the video to.

        Returns:
        dict: A dictionary containing the metadata of the video.
        """
        try:
            ydl_opts = {
                'format': 'best',
                'outtmpl': f'{output_path}/input_vid.%(ext)s',
                'quiet': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                ext = info_dict.get("ext")
                metadata = {
                    "Author": info_dict.get('uploader'),
                    "Title": info_dict.get('title'),
                    "Views": info_dict.get('view_count')
                }
                metadata = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
            logger.info("Video downloaded successfully.")
            return metadata, f'{output_path}/input_vid.%(ext)s'.replace("%(ext)s", ext)
        except Exception as e:
            logger.error(f"Failed to download video. An error occurred: {e}")
            return None

    def video_to_images(self, video_path: str, output_folder: str):
        try:
            clip = VideoFileClip(video_path)
            os.makedirs(output_folder, exist_ok=True)
            fps = self.config.video_fps
            duration = int(clip.duration)
            timestamps = {}

            for t in range(0, int(duration * fps)):
                frame_time = t / fps
                frame = clip.get_frame(frame_time)
                frame_path = os.path.join(output_folder, f"frame_{t:05d}.png")
                ImageClip(frame).save_frame(frame_path)
                timestamps[f"frame_{t:05d}.png"] = {"filename": frame_path, "timestamp": frame_time}

            logger.info("Frames extracted successfully.")
            return timestamps
        except Exception as e:
            logger.error(f"Failed to extract frames. An error occurred: {e}")
            return None

    @staticmethod
    def video_to_audio(video_path, output_audio_path):
        """
        Convert a video to audio and save it to the output path.

        Parameters:
        video_path (str): The path to the video file.
        output_audio_path (str): The path to save the audio to.

        """
        logger.info("Start converting video to audio...")
        clip = VideoFileClip(video_path)
        audio = clip.audio
        audio.write_audiofile(output_audio_path)
        logger.info("Convert video to audio successfully")

    @staticmethod
    def audio_to_text(audio_path):
        """
        Convert an audio file to text.

        Parameters:
        audio_path (str): The path to the audio file.

        Returns:
        test (str): The text recognized from the audio.

        """
        recognizer = sr.Recognizer()
        audio = sr.AudioFile(audio_path)
        logger.info("Converting audio to text")

        with audio as source:
            # Record the audio data
            audio_data = recognizer.record(source)

            try:
                # Recognize the speech
                text = recognizer.recognize_whisper(audio_data)
            except sr.UnknownValueError:
                logger.error("Speech recognition could not understand the audio.")
            except sr.RequestError as e:
                logger.error(f"Could not request results from service; {e}")

        return text

    def process_video(
        self,
        filepath: str,
        output_folder: str,
        output_frames_path: str,
        output_audio_path: str,
    ):
        timestamps = self.video_to_images(filepath, output_frames_path)
        self.video_to_audio(filepath, output_audio_path)
        text_data = self.audio_to_text(output_audio_path)

        with open(output_folder + "output_text.txt", "w") as file:
            file.write(text_data)
        logger.info("Text data saved to file")
        return timestamps

    def __call__(self, url: str, output_folder: str):
        output_video_path = output_folder + "/video_data/"
        output_audio_path = output_folder + "/mixed_data/output_audio.wav"
        os.makedirs(output_video_path, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.split(output_audio_path)[0], exist_ok=True)

        metadata, filepath = self.download_video(url=url, output_path=output_video_path)
        timestamps = self.process_video(
            filepath=filepath,
            output_folder=output_folder,
            output_frames_path=output_folder,
            output_audio_path=output_audio_path
        )
        self.metadata = metadata
        self.timestamps = timestamps
        logger.info("Process video done!")
