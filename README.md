# Question Answering Chatbot with Video

## Overview

- This is a simple Chatbot that can help user to QA about information of a video.
- For the current version:
    - chatbot can only receive Youtube video url as input.
    - just allow user to input message as text, not image yet
- Chatbot using GPT4o as base LLM.

### Process flow:

- This chatbot basically use idea of RAG, the pipeline has the following steps:
    - First: It will ask user to input a Youtube video url
        - Video Processor:
            - After receive the url, it will download it to the temporal directory using `tempfile` lib of python.
            - Then it will convert the video into images (it's also base on the video fps of your setting) and convert
              video
              audio into text.
        - Retriever Processor:
            - This processor will receive the folder path of image frames and audio, and it will index the data.
    - Second: After processing the video (it might take you some minutes to finish), user now can input text message and
      ask about the information in the video.

## Installation

- Install required packages: `pip install -r requirements.txt`
- Install `ffmpeg`: `conda install conda-forge::imageio-ffmpeg`

## Running

- For the current version, it is highly recommended that you should running with Terminal, because gradio app may still
  have some bugs, sorry for this inconvenience.

### Terminal

- You can test the bot in terminal by running: `python launch.py`
- You also have to set the OPENAI_API_KEY env variable to use gpt4o

### Gradio

- Also, you can run the chatbot with UI by running: `python gradio_app.py`

## Proposal

- For current version, the chatbot does not handle images as input message, but in I think we can implement this feature
  by:
    - Step 1: With each input message, use gpt4o to generate the summarize for that image.
    - Step 2: Use that summarized text to and retrieve a list of relevant images using retriever processor.
    - Step 3: Use these retrieved images + input text of user and input it to gpt4o, and return to user the response.
  