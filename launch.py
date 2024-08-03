import tempfile

from processors import VideoProcessor, Retriever, Config
from processors.processor import ConversationBot
from processors.llms import GPT4o

if __name__ == "__main__":
    config = Config(output_folder="temp_data", video_fps=1, max_new_tokens=1500)
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
    print(f"Reading video from url: {url} ...")
    bot.read_video(url=url)

    print("Indexing data ...")
    bot.index(data_path=output_folder.name)

    while True:
        user_msg = input("User: ")
        if not user_msg:
            continue
        response = bot.chat(user_message=user_msg)
        print(f"\033[92mBot: {response}\n\033[00m")
