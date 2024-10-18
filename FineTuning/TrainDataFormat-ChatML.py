import os
import json

# 训练数据集文件
BAS_DATA_DIR = 'D:\ModelSpace\Qwen2'
RAW_TRAIN_FILE_PATH = os.path.join(BAS_DATA_DIR, os.path.join('zh_cls_fudan-news', 'train.jsonl'))
NEW_TRAIN_FILE_PATH = os.path.join(BAS_DATA_DIR, 'train-ChatML.jsonl')

# 将原始数据集转换为ChatML格式的新数据集
message_list = []

# 读取原JSONL文件
with open(RAW_TRAIN_FILE_PATH, "r", encoding="utf-8") as file:
	for line in file:
		# 解析每一行原始数据（每一行均是一个JSON格式）
		data = json.loads(line)
		text = data["text"]
		catagory = data["category"]
		output = data["output"]

		message_part_1 = {"role": "system", "content": "You are a helpful assistant"}
		message_part_2 = {"role": "user", "content": f"你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项列表，请输出文本内容的正确分类。\n{text}\n分类选项列表:{catagory}"}
		message_part_3 = {"role": "assistant", "content": output}
		message = {
			"messages": [message_part_1, message_part_2, message_part_3]
		}

		message_list.append(message)

# 保存处理后的JSONL文件，每行也是一个JSON格式
with open(NEW_TRAIN_FILE_PATH, "w", encoding="utf-8") as file:
	for message in message_list:
		file.write(json.dumps(message, ensure_ascii=False) + "\n")
