curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "/home/obullxl/ModelSpace/Qwen2-0.5B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "天空为什么是蓝色的？"}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "repetition_penalty": 1.05,
  "max_tokens": 512
}'

