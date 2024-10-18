import ChatTTS
import torch
import torchaudio
from tools.logger import get_logger
from pydub import AudioSegment
import os

logger = get_logger("SunningTX")

#
# 第一步：ChatTTS初始化
#
# 1. 初始化ChatTTS实例
chat = ChatTTS.Chat(
    get_logger("ChatTTS"),
)

# 2. 本地下载模型文件，需要魔改文件摘要，否则模型加载失败
chat.sha256_map = {
    "sha256_asset_Decoder_pt": "9964e36e840f0e3a748c5f716fe6de6490d2135a5f5155f4a642d51860e2ec38",
    "sha256_asset_DVAE_full_pt": "553eb75763511e23f3e5f86303e2163c5ca775489d637fb635d979c8ae58bbe5",
    "sha256_asset_GPT_pt": "d7d4ee6461ea097a2be23eb40d73fb94ad3b3d39cb64fbb50cb3357fd466cadb",
    "sha256_asset_Vocos_pt": "09a670eda1c08b740013679c7a90ebb7f1a97646ea7673069a6838e6b51d6c58",
    "sha256_asset_tokenizer_special_tokens_map_json": "30e1f3dfbaef18963d9298e9a70e5f4be9017b4f64ec67737d0bccc2dbeba8c9",
    "sha256_asset_tokenizer_tokenizer_config_json": "ac0eb91b587bd4c627c2b18ac1652bdf3686c7a6cd632d4b00feb9f34828dfdc",
    "sha256_asset_tokenizer_tokenizer_json": "dddc3d54016d6cb75ed66dde3be50287afe6dee679c751538373feb75f950020"
}

# 3. 加载模型文件：指定模型文件路径
chat.load(
    compile=True,
    source='custom',
    force_redownload=False,
    custom_path='D:\ModelSpace\ChatTTS-Model',
)

#
# 第二步：按行准备故事文本内容，并转换为音频文件
#
# 1. 按行读取故事内容（·《》：符号无法识别，需要过滤掉）
lines = []
with open('./Story.txt', 'r', encoding='utf-8') as file:
    for line in file:
        # 去掉或者替换不支持的字符
        line = line.replace('·', ' ')
        line = line.replace('《', '').replace('》', '')
        line = line.replace('：', ' ')

        # 最后末尾增加一个空格，以暂停一下
        line = line.strip() + ' '

        if len(lines) > 2:
            pass

        if len(line) > 0:
            lines.append(line)
            logger.info(line)

# 2. 音频转换，按照每行文本转换为一个音频文件
temp_dir = os.path.join(os.getcwd(), 'temp')
os.makedirs(temp_dir, exist_ok=True)


# 存在单个音频文件
def save_wav(idx: int, src: torch.Tensor):
    wav_path = f'./temp/{idx}.wav'
    try:
        torchaudio.save(
            wav_path,
            torch.from_numpy(src).unsqueeze(0),
            24000,
            format='wav',
            backend='soundfile'
        )
    except:
        torchaudio.save(
            wav_path,
            torch.from_numpy(src),
            24000,
            format='wav',
            backend='soundfile'
        )

    return wav_path


# 逐行合成音频文件
wav_list = []
for i in range(len(lines)):
    line = lines[i]
    logger.info(f'合成音频：{line}')

    res_gen = chat.infer(
        lines[i],
        skip_refine_text=False,
        refine_text_only=False,
        params_refine_text=ChatTTS.Chat.RefineTextParams(
            temperature=0.3,
            top_P=0.7,
            top_K=20,
            manual_seed=222,
        ),
    )
    wav_list.append(save_wav(i, res_gen))


#
# 第三步：合并所有短音频为长音频文件
#
def merge_wav_files(input_files, output_file):
    combined_audio = AudioSegment.empty()

    for input_file in input_files:
        audio = AudioSegment.from_wav(input_file)
        combined_audio += audio

    combined_audio.export(output_file, format='wav')


# 合并音频文件
merge_wav_files(wav_list, './Story.wav')
