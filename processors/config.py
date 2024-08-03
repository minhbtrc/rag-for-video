import os


class Config:
    def __init__(
        self,
        output_folder: str,
        video_fps: float = 0.02,
        max_new_tokens: int = 1500,
        openai_api_key: str = None
    ):
        self.output_folder = output_folder
        self.video_fps = video_fps
        self.max_new_tokens = max_new_tokens
        self.OPENAI_API_TOKEN = openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_TOKEN")
