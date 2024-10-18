#!/usr/bin/env python
# encoding: utf-8

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# 模型权重文件目录
model_dir = '.'

# 加载模型：local_files_only 加载本地模型，trust_remote_code 执行远程代码（必须）
model = AutoModel.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=True,
)

# 设置推理模式，如果有卡：model = model.eval().cuda()
model = model.eval()

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=True,
)

# 测试的汽车尾部图片，可以指定其它目录
image = Image.open('Car-01.jpeg').convert('RGB')

# 图片理解：自然语言理解 + 图片理解
question = '请问这是一张什么图片？'
msgs = [{'role': 'user', 'content': [image, question]}]

res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    stream=True
)

# 理解结果
generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')
