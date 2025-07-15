import os
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
import requests

client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-de37c002186e4c3dbb84dbe9a1459893",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

audio_path = "C:/Users/Administrator/Desktop/test.wav"
audio_data = encode_audio(audio_path)


completion = client.chat.completions.create(
    model="qwen-omni-turbo-0119",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:;base64,{audio_data}",
                        "format": "wav",
                    },
                },
                {"type": "text", "text": "请分析这段音频，判断其中是否只包含单一音源或单一音频事件。如果音频中只有一种声音（例如只有狗叫声、只有人说话声或只有单一乐器演奏等），则判断为'单源音频'；如果音频中同时包含多种不同的声音（例如同时有猫叫声和音乐声、人说话和汽车噪音等混合在一起），则判断为'多源音频'。请详细说明你的判断理由，并列出你在音频中识别到的所有声音类型。"},
            ],
        },
    ],
    # 设置输出数据的模态，当前支持两种：["text","audio"]、["text"]
    modalities=["text"],
    # audio={"voice": "Cherry", "format": "wav"},
    # stream 必须设置为 True，否则会报错
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in completion:
    if chunk.choices:
        print(chunk.choices[0].delta)
    else:
        print(chunk.usage)
