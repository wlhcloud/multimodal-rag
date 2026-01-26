
from sentence_transformers import SentenceTransformer


# 指定本地模型路径
model_path = "/home/gybwg/ai-project/models/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"  # 请替换为你的实际路径

# 方式1: 直接指定模型名并设置 cache_folder（如果模型已下载到本地，且结构符合 sentence-transformers 的预期）
gme_st = SentenceTransformer(
    model_path, # 或者使用本地路径 model_path
)


corpus_texts = [
    "图2 中东地区陶瓷釉的$Fe/Co$和$Mn/Co$",
]

# 假设的图像数据路径列表（本地路径）
corpus_images = [
    "/home/gybwg/ai-project/project/Multimodal_RAG/output/images/06deb205fbda705dfc5a1d96fae0cdae.png"
    "/home/gybwg/ai-project/project/Multimodal_RAG/output/images/06deb205fbda705dfc5a1d96fae0cdae.png"
    "图2 中东地区陶瓷釉的$Fe/Co$和$Mn/Co$",
]


# 1. 编码文本
# 对于文本，可以直接传入字符串列表
text_embeddings = gme_st.encode(corpus_texts, convert_to_tensor=True) # 得到文本向量
print("文本向量形状:", text_embeddings.shape)


# 2. 编码图像
# 对于图像，GME模型通常需要以特定格式传入，例如字典表明类型
# 根据 GME 模型的预期输入格式，可能需要将图像路径包装成字典
image_inputs = [img_path for img_path in corpus_images]
image_embeddings = gme_st.encode(image_inputs, convert_to_tensor=True) # 得到图像向量
print("图像向量形状:", len(image_embeddings[0].tolist()))


