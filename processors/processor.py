from typing import TYPE_CHECKING

from llama_index.core import SimpleDirectoryReader

if TYPE_CHECKING:
    from processors.video import VideoProcessor
    from processors.retriever import Retriever
    from config import Config
    from processors.llms.base import LLM


class ConversationBot:
    def __init__(
        self,
        config: "Config",
        video_processor: "VideoProcessor",
        retriever_processor: "Retriever",
        database_path: str,
        llm: "LLM"
    ):
        self.config = config
        self.video_processor = video_processor
        self.retriever_processor = retriever_processor
        self.database_path = database_path
        self.llm = llm
        self._metadata = ""

    @property
    def prompt(self):
        return """Given the provided information, including relevant images and retrieved context from the video, 
        accurately and precisely answer the query without any additional prior knowledge.

Please ensure honesty and responsibility, refraining from any racist or sexist remarks.

---------------------

Context: {context}

Metadata for video: {metadata}

---------------------

Query: {query}

Answer:"""

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, _metadata):
        self._metadata = _metadata

    def read_video(self, url: str, ):
        metadata = self.video_processor(url=url, output_folder=self.database_path)

    def retrieve_relevant_info(self, query_str: str):
        img, txt = self.retriever_processor.retrieve(query_str=query_str)
        image_documents = SimpleDirectoryReader(
            input_dir=self.database_path, input_files=img, file_metadata=self.video_processor.get_timestamps
        ).load_data(show_progress=True)
        return "".join(txt), image_documents

    def index(self, data_path: str):
        self.retriever_processor.index_data(output_folder=data_path)

    def chat(self, user_message: str) -> str:
        contexts, image_documents = self.retrieve_relevant_info(query_str=user_message)
        context_str = "".join(contexts)
        metadata_str = self.video_processor.metadata
        prompt = self.prompt.format(
            context=context_str,
            query=user_message,
            metadata=metadata_str
        )

        response = self.llm.generate(prompt=prompt, images=image_documents)
        return response
