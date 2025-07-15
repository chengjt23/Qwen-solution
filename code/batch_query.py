import os
from openai import OpenAI
import base64
import json

def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")
    
def access_ai(audio_path):
    client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-de37c002186e4c3dbb84dbe9a1459893",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
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
                            "format": "flac",
                        },
                    },
                    {"type": "text", "text": "请分析这段音频，判断其中是否只包含一种类型的声音。请只返回True或者False，True表示该音频只包含一种类型的声音，False表示该音频包含多种类别的声音。下面是你的回复："},
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
    # 收集完整的响应内容
    full_response = ""
    usage_info = None

    for chunk in completion:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        else:
            usage_info = chunk.usage

    return full_response

def check_single(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    result = []
    for item in data:
        audio_path = item["audio_path"]

        response = access_ai(audio_path)
        # 检测返回值并在需要时重试
        retry_count = 0
        max_retries = 3
        while not (response.startswith("True") or response.startswith("False")):
            response = access_ai(audio_path)
            retry_count += 1
            if retry_count >= max_retries:
                print(f"达到最大重试次数，音频路径: {audio_path}")
                result.append("None")
                break
            
        result.append("True" if response.startswith("True") else "False")
    return result

if __name__ == "__main__":
    data_path = "/gpfs-flash/hulab/public_datasets/audio_datasets/UAS_Benchmark/other_dataset/Qwen-solution/data/test_data.json"
    result = check_single(data_path)
    print(result)