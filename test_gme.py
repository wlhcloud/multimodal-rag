
from sentence_transformers import SentenceTransformer


# 指定本地模型路径
model_path = "/home/gybwg/ai-project/models/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"  # 请替换为你的实际路径

# 方式1: 直接指定模型名并设置 cache_folder（如果模型已下载到本地，且结构符合 sentence-transformers 的预期）
model = SentenceTransformer(
    # 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct', # 或者使用本地路径 model_path
    cache_folder=model_path
)


corpus_texts = [
    "Lambda架构中针对实时数据处理我们可以使用Spark计算框架进行分析,Spark针对实时数据进行分析本质是将实时流数据看成微批进行处理。",
    "基于有状态计算的方式最大的优势是不需要将原始数据重新从外部存储中拿出来,从而进行全量计算,因为这种计算方式的代价可能是非常高的。",
]

# 假设的图像数据路径列表（本地路径）
corpus_images = [
    "/mnt/d/ProdProject/Python/ModelTuneAi/Multimodal_RAG/output/images"
]


# 1. 编码文本
# 对于文本，可以直接传入字符串列表
text_embeddings = model.encode(corpus_texts, convert_to_tensor=True) # 得到文本向量
print("文本向量形状:", text_embeddings.shape)


# 2. 编码图像
# 对于图像，GME模型通常需要以特定格式传入，例如字典表明类型
# 根据 GME 模型的预期输入格式，可能需要将图像路径包装成字典
image_inputs = [{"image": img_path} for img_path in corpus_images]
image_embeddings = model.encode(image_inputs, convert_to_tensor=True) # 得到图像向量
print("图像向量形状:", image_embeddings.shape)


