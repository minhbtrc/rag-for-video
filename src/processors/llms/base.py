import abc


class LLM(abc.ABC):
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def generate(self, prompt: str, images: list):
        pass

    @abc.abstractmethod
    async def agenerate(self, prompt: str, images: list):
        pass
