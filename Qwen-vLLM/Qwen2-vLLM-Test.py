import os
from vllm import LLM, SamplingParams

# 设置环境变量
os.environ['VLLM_TARGET_DEVICE'] = 'cpu'
# os.environ['VLLM_CPU_KVCACHE_SPACE'] = '10'

# 模型ID：我们下载的模型权重文件目录
model_dir = '/home/obullxl/ModelSpace/Qwen2-0.5B'

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(
	model=model_dir,
	device='cpu',
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

