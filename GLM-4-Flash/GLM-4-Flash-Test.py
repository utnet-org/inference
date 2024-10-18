# GLM-4-Flash-Test.py
from zhipuai import ZhipuAI
import os

# 环境变量中获取API Key内容
client = ZhipuAI(
  api_key=os.environ.get('ZhipuAI-APIKey')
)

# 使用GLM-4-Flash大模型进行验证
response = client.chat.completions.create(
  model='glm-4-flash',
  messages=[
    {'role': 'user', 'content': '你好，我是老牛同学，请问你是谁？'},
  ],
  stream=True,
)

# 流式输出
for chunk in response:
    print(chunk.choices[0].delta.content, end='')

print('')
