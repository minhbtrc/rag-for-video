import base64
import os
import json
import tempfile

import gradio as gr
from llama_index.core import StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from processors import VideoProcessor, Retriever, Config
from processors.processor import ConversationBot
from processors.llms import GPT4o

log_to_console = False

temp_files = []


class App:
    def __init__(self):
        config = Config(output_folder="temp_data", video_fps=0.2, max_new_tokens=1500)
        output_folder = tempfile.TemporaryDirectory(dir=config.output_folder)

        video_processor = VideoProcessor(config=config)
        self.retriever = Retriever(config=config)
        llm = GPT4o(config=config)

        self.bot = ConversationBot(
            video_processor=video_processor,
            retriever_processor=self.retriever,
            config=config,
            database_path=output_folder.name,
            llm=llm
        )
        self.bot = None

    def encode_image(self, image_data):
        """Generates a prefix for image base64 data in the required format for the
        four known image formats: png, jpeg, gif, and webp.

        Args:
        image_data: The image data, encoded in base64.

        Returns:
        A string containing the prefix.
        """

        # Get the first few bytes of the image data.
        magic_number = image_data[:4]

        # Check the magic number to determine the image type.
        if magic_number.startswith(b'\x89PNG'):
            image_type = 'png'
        elif magic_number.startswith(b'\xFF\xD8'):
            image_type = 'jpeg'
        elif magic_number.startswith(b'GIF89a'):
            image_type = 'gif'
        elif magic_number.startswith(b'RIFF'):
            if image_data[8:12] == b'WEBP':
                image_type = 'webp'
            else:
                # Unknown image type.
                raise Exception("Unknown image type")
        else:
            # Unknown image type.
            raise Exception("Unknown image type")

        return f"data:image/{image_type};base64,{base64.b64encode(image_data).decode('utf-8')}"

    def encode_file(self, fn: str) -> list:
        user_msg_parts = []

        with open(fn, mode="rb") as f:
            content = f.read()

        if isinstance(content, bytes):
            try:
                # try to add as image
                content = self.encode_image(content)
            except:
                # not an image, try text
                pass
        else:
            content = str(content)

        user_msg_parts.append({"type": "image_url",
                               "image_url": {"url": content}})

        return user_msg_parts

    def undo(self, history):
        history.pop()
        return history

    def dump(self, history):
        return str(self.history)

    def load_settings(self):
        # Dummy Python function, actual loading is done in JS
        pass

    def save_settings(self, openai_api_key, video_url, tokens: int):
        print(88888888)
        # Dummy Python function, actual saving is done in JS
        config = Config(
            output_folder="temp_data",
            video_fps=0.2,
            max_new_tokens=tokens,
            openai_api_key=openai_api_key,
        )
        output_folder = tempfile.TemporaryDirectory(dir=config.output_folder)

        video_processor = VideoProcessor(config=config)
        retriever = Retriever(config=config)
        llm = GPT4o(config=config)

        self.bot = ConversationBot(
            video_processor=video_processor,
            retriever_processor=retriever,
            config=config,
            database_path=output_folder.name,
            llm=llm
        )
        self.bot.read_video(url=video_url)
        self.bot.index(data_path=output_folder.name)

    SYS_PROMPT = ""

    def format_messages(self, history: list):
        return "\n".join([f"{ele['role']}: {ele['content']}" for ele in history])

    def main(self, message, history, oai_key, video_url, max_tokens):
        try:
            if log_to_console:
                print(f"bot history: {str(history)}")

            history_openai_format = []
            user_msg_parts = []
            history_openai_format.append({"role": "system", "content": self.SYS_PROMPT})
            for human, assi in history:
                if human is not None:
                    if type(human) is tuple:
                        user_msg_parts.extend(self.encode_file(human[0]))
                    else:
                        user_msg_parts.append({"type": "text", "text": human})

                if assi is not None:
                    if user_msg_parts:
                        history_openai_format.append({"role": "user", "content": user_msg_parts})
                        user_msg_parts = []

                    history_openai_format.append({"role": "assistant", "content": assi})

            if message['text']:
                user_msg_parts.append({"type": "text", "text": message['text']})
            if message['files']:
                for file in message['files']:
                    user_msg_parts.extend(self.encode_file(file['path']))
            history_openai_format.append({"role": "user", "content": user_msg_parts})

            if log_to_console:
                print(f"br_prompt: {str(history_openai_format)}")

            # response = client.chat.completions.create(
            #     messages=history_openai_format,
            #     max_tokens=max_tokens
            # )
            # TODO: generate response, chat function receive input as text
            response = 1  # bot.chat(user_message=format_messages(history_openai_format))

            if log_to_console:
                print(f"br_response: {str(response)}")

            # result = response.choices[0].message.content

            if log_to_console:
                print(f"br_result: {str(history)}")

        except Exception as e:
            raise gr.Error(f"Error: {str(e)}")

        return response

    def import_history(self, history, file):
        with open(file.name, mode="rb") as f:
            content = f.read()

            if isinstance(content, bytes):
                content = content.decode('utf-8', 'replace')
            else:
                content = str(content)
        os.remove(file.name)

        # Deserialize the JSON content
        import_data = json.loads(content)

        # Check if 'history' key exists for backward compatibility
        if 'history' in import_data:
            history = import_data['history']
        else:
            # Assume it's an old format with only history data
            history = import_data

        return history  # Return system prompt value to be set in the UI

    def start_demo(self, host="localhost", port=8000, debug=False, share=True):
        with gr.Blocks(delete_cache=(86400, 86400)) as demo:
            gr.Markdown("# Question Answering with Video")
            # with gr.Accordion("Settings"):
            model = gr.Dropdown(
                label="Model",
                value="gpt-4o",
                allow_custom_value=True,
                elem_id="model",
                choices=["gpt-4o"]
            )
            oai_key = gr.Textbox(label="OpenAI API Key", elem_id="oai_key", value="!313")
            video_url = gr.Textbox(label="Youtube Video URL", elem_id="video_url", value="1313")
            max_tokens = gr.Slider(1, 4000, label="Max. Tokens", elem_id="max_tokens", value=1500)
            save_button = gr.Button("Save Settings")

            save_button.click(self.save_settings, [oai_key, video_url, max_tokens], show_progress=True)
            controls = [oai_key, video_url, max_tokens]

            chat = gr.ChatInterface(fn=self.main, multimodal=True, additional_inputs=controls)
            chat.textbox.file_count = "multiple"
            chatbot = chat.chatbot
            chatbot.show_copy_button = True
            chatbot.height = 500

        # demo.unload(lambda: [os.remove(file) for file in temp_files])
        demo.queue()
        demo.launch(debug=debug, server_port=port, share=share)


if __name__ == "__main__":
    app = App()
    app.start_demo(port=8000, debug=True, share=False)
