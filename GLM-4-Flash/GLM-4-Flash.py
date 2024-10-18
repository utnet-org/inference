# GLM-4-Flash.py
from zhipuai import ZhipuAI
import os
import requests


# 构造客户端
def make_client():
  return ZhipuAI(
    api_key=os.environ.get('ZhipuAI-APIKey')
  )


#
# 第一步：使用GLM-4-Flash大模型产出创意
#
def make_idea():
  response = make_client().chat.completions.create(
    model='glm-4-flash',
    messages=[
      {'role': 'system', 'content': '你是一位儿童绘本的内容创意专家，你的任务是根据用户提供的主题，提供适合7岁到10岁小学生阅读的、专业的、有见地的绘本内容创意。'},
      {'role': 'user', 'content': '请以“黑神话·悟空”这款最近热门的游戏为主题，提供儿童绘本创意。要求：绘本分为4个小段，每个小段需要有插图。'},
    ],
    stream=True,
  )

  # 流式输出
  idea = ''
  for chunk in response:
      idea += chunk.choices[0].delta.content

  return idea


#
# 第二步：使用GLM-4-Flash大模型生成故事内容
#
def make_content(idea:str):
  response = make_client().chat.completions.create(
    model='glm-4-flash',
    messages=[
      {'role': 'system', 'content': '你是一位儿童绘本的故事内容编写专家，你编写的故事幽默有趣，特别适合7岁到10岁的小学生阅读，你的任务是根据用户提供的儿童绘本创意，完成编写的整个故事内容。'},
      {'role': 'user', 'content': f'请根据儿童绘本创意，完成编写整个故事内容。\n\n故事内容要求：\n故事内容分为4个小段，每个小段500个汉字左右，故事总长度不得超过2000个汉字。\n\n儿童绘本创意：\n{idea}'},
    ],
    stream=True,
  )

  # 流式输出
  content = ''
  for chunk in response:
      content += chunk.choices[0].delta.content

  return content

#
# 第三步：使用CogView-3大模型生成故事插图
#

# 存储图片到本地
def download_image(url, save_path):
  print(f'开始下载图片: {url}')
  
    # 发送 HTTP 请求
  response = requests.get(url, stream=True)

    # 检查请求是否成功
  if response.status_code == 200:
    # 以二进制模式打开文件
    with open(save_path, 'wb') as file:
      # 将图片内容写入文件
      file.write(response.content)
    print(f'图片下载成功: {save_path}')
  else:
    print('图片下载失败.')


# 生成图片
def make_illustration(idea:str):
  item_list = ['第一段', '第二段', '第三段', '第四段']

  for item in item_list:
    response = make_client().images.generations(
        model='cogview-3',
        prompt=f'你是一位儿童绘本插图绘画专家，你画的插图紧贴绘本的创意，插图色彩鲜艳，画面生动，有助于培养7岁到10岁的小学生的审美观。\n\n下面是一个儿童绘本的创意，共有4个小段，请为“{item}”画一张插图：\n\n{idea}'
    )

    image_url = response.data[0].url

    print(f'{item}插图地址：{image_url}')

    # 图片目录
    image_dir = os.path.join(os.getcwd(), 'assets')
    os.makedirs(image_dir, exist_ok=True)

    # 图片路径
    save_path = os.path.join(image_dir, f'{item}.jpg')

    # 下载存储图片
    download_image(image_url, save_path)



# 运行
if __name__ == '__main__':
  # 1. 绘本构思
  print('----------------------------- STRT：绘本构思 -----------------------------')
  idea = make_idea()
  print(idea)
  print('----------------------------- 绘本构思：END ------------------------------')

  # 2. 绘本内容
  print('----------------------------- STRT：绘本内容 -----------------------------')
  content = make_content(idea)
  print(content)
  print('----------------------------- 绘本内容：END ------------------------------')

  # 3. 绘本插图
  print('----------------------------- STRT：绘本插图 -----------------------------')
  make_illustration(idea)
  print('----------------------------- 绘本插图：END ------------------------------')

