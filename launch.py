import tempfile

from src.processors import VideoProcessor, Retriever, Config
from src.processors.processor import ConversationBot
from src.processors.llms import GPT4o

config = Config(output_folder="temp_data", video_fps=2, max_new_tokens=1500)
output_folder = tempfile.TemporaryDirectory(dir=config.output_folder)

video_processor = VideoProcessor(config=config)
retriever = Retriever(config=config)
llm = GPT4o(config=config)

bot = ConversationBot(
    video_processor=video_processor,
    retriever_processor=retriever,
    config=config,
    database_path=output_folder.name,
    llm=llm
)

url = None
while not url:
    url = input("Please input video url:")
    if url == "exit":
        import sys

        sys.exit()
# "https://www.youtube.com/watch?v=9RhWXPcKBI8"
print(f"Reading video from url: {url} ...")
bot.read_video(url=url)

print("Indexing data ...")
bot.index(data_path=output_folder.name)

while True:
    user_msg = input("User: ")
    if not user_msg:
        continue
    response = bot.chat(user_message=user_msg)
    print(f"Bot: {response}\n\n")
