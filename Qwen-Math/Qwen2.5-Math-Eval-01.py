#
# conda create -n Qwen2.5 python=3.10 -y
#
# conda activate Qwen2.5
#
# pip install torch
# pip install modelscope
# pip install "transformers>=4.37.0"
# pip install "accelerate>=0.26.0"
#

import os
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_dir = os.path.join('D:', os.path.sep, 'ModelSpace', 'Qwen2.5', 'Qwen2.5-Math-1.5B-Instruct')
print(f'权重目录: {model_dir}')

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype='auto',
    device_map='auto',
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True,
)

prompt = '请计算等式中的X值: 4X+5=6X+7'
messages = [
    {'role': 'system', 'content': '你是一位数学专家，特别擅长解答数学题。'},
    {'role': 'user', 'content': prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer(
    [text],
    return_tensors='pt',
).to(model.device)

print(f'开始推理: {prompt}')

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
)

print('推理完成.')

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]

print(f'推理结果: {response}')
