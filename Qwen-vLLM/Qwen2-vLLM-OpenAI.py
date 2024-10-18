from openai import OpenAI

# OpenAI初始化
client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)

chat_response = client.chat.completions.create(
    model='/home/obullxl/ModelSpace/Qwen2-0.5B',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '天空为什么是蓝色的？'},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
)

print('Qwen2推理结果:', chat_response)

