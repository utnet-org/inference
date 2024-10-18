import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 设置环境变量
os.environ['VLLM_TARGET_DEVICE'] = 'cpu'

# 模型ID：我们下载的模型权重文件目录
model_dir = '/home/obullxl/ModelSpace/Qwen2-0.5B'

# Tokenizer初始化
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True,
)

# Prompt提示词
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '天空为什么是蓝色的？'}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# 初始化大语言模型
llm = LLM(
    model=model_dir,
    tensor_parallel_size=1,  # CPU无需张量并行
    device='cpu',
)

# 超参数：最多512个Token
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# 模型推理输出
outputs = llm.generate([text], sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text

    print(f'{generated_text!r}')

