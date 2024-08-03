from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from processors.llms.base import LLM


class GPT4o(LLM):
    def __init__(self, config):
        super().__init__(config=config)
        self.model = OpenAIMultiModal(
            model="gpt-4o", api_key=self.config.OPENAI_API_TOKEN, max_new_tokens=self.config.max_new_tokens
        )

    def generate(self, prompt: str, images: list):
        response = self.model.complete(
            prompt=prompt,
            image_documents=images
        )
        return response.text

    async def agenerate(self, prompt: str, images: list):
        pass
