#!/usr/bin/env python
# encoding: utf-8

import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
from decord import VideoReader, cpu

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


MAX_NUM_FRAMES=64

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

# 视频文件路径
video_path="~/Car.mp4"

frames = encode_video(video_path)

question = "请问这是一个什么视频？"
msgs = [
    {'role': 'user', 'content': frames + [question]},
]

# Set decode params for video
params={}
params["use_image_id"] = False
params["max_slice_nums"] = 2 # 如果cuda OOM且视频分辨率大于448*448 可设为1

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    **params
)

print(answer)
