from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-Embedding-0.6B",
    cache_dir="/mnt/d/ProdProject/AI/models",
    revision="master",
)
# from datasets import load_dataset
# load_dataset("nlhappy/DuIE")
# import torch
#
# # 检查CUDA是否可用
# print(f"CUDA available: {torch.cuda.is_available()}")  # True/False
#
# # 检查当前使用的设备
# print(f"Current device: {torch.cuda.current_device()}")  # GPU索引
#
# # 查看GPU信息（如果可用）
# if torch.cuda.is_available():
#     print(f"GPU Name: {torch.cuda.get_device_name(0)}")
#     print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
# else:
#     print("PyTorch is using CPU")


# from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
# from datasets import load_dataset
