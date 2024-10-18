# Qwen2.5-Math-奥数推理.py

import os
import json
from openai import OpenAI

# 初始化客户端：提前配置好环境变量
client = OpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
)

# 读取奥数题目
input_file = 'Qwen2.5-Math-奥数题目.json'

with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 循环每道题目，请求推理服务
output_data = []

for item in data:
    print('')
    print(f'奥数题目-> {item["title"]}')
    print(f'期望答案-> {item["answer"]}')

    completion = client.chat.completions.create(
        model='qwen2.5-math-72b-instruct',
        messages=[
            {'role': 'system', 'content': '你是一位数学专家，特别擅长解答数学题。'},
            {'role': 'user', 'content': item["title"]}
        ],
    )

    # 获取推理结果
    result = json.loads(completion.model_dump_json())
    content = result['choices'][0]['message']['content']

    print(f'推理结果-> {content}')

    output = {
        'level': item['level'],
        'title': item['title'],
        'answer': item['answer'],
        'result': content
    }

    output_data.append(output)

    print('')

# 保存推理结果
output_file = 'Qwen2.5-Math-推理结果.md'

with open(output_file, "w", encoding="utf-8") as file:
    for output in output_data:
        file.write(f'{output["level"]}题目：{output["title"]}\n\n期望答案：{output["answer"]}\n\n')
        file.write(f'{output["result"]}\n\n---\n\n')
