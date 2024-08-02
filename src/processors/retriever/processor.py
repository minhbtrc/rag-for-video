from llama_index.vector_stores.lancedb import LanceDBVectorStore
from loguru import logger
from typing import TYPE_CHECKING

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
# from llama_index.legacy import StorageContext
# from llama_index.legacy.vector_stores import LanceDBVectorStore

if TYPE_CHECKING:
    from src.processors.config import Config


class Retriever:
    def __init__(self, config: "Config"):
        text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
        image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
        self.storage_context = StorageContext.from_defaults(
            vector_store=text_store,
            image_store=image_store
        )
        self.retriever_engine = None

    def index_data(self, output_folder: str):
        logger.info("Indexing data ...")
        documents = SimpleDirectoryReader(output_folder).load_data()

        index = MultiModalVectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
        )
        self.retriever_engine = index.as_retriever(
            similarity_top_k=5, image_similarity_top_k=5
        )

    def retrieve(self, query_str):
        retrieval_results = self.retriever_engine.retrieve(query_str)

        retrieved_image = []
        retrieved_text = []
        for res_node in retrieval_results:
            if isinstance(res_node.node, ImageNode):
                retrieved_image.append(res_node.node.metadata["file_path"])
            else:
                display_source_node(res_node, source_length=200)
                retrieved_text.append(res_node.text)

        return retrieved_image, retrieved_text
