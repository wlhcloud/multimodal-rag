from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from zai import ZhipuAiClient

from utils.env_utils import ALIBABA_API_KEY, ALIBABA_BASE_URL, ZHIPU_API_KEY, LOCAL_GME_MODEL_PATH, LOCAL_TEXT_EMB_PATH


class CustomQwen3Embeddings(Embeddings):
    """自定义一个qwen3的Embedding和langchain整合的类"""

    def __init__(self, model_name):
        self.qwen3_embedding = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> ndarray:
        return self.qwen3_embedding.encode(texts)


multiModal_llm = ChatOpenAI(# 多模态大模型
    model='qwen3-vl-plus',
    api_key=ALIBABA_API_KEY,
    base_url=ALIBABA_BASE_URL,
)

llm = ChatOpenAI(# 多模态大模型
    model='qwen-vl-plus',
    api_key=ALIBABA_API_KEY,
    base_url=ALIBABA_BASE_URL,
)

embedding = OpenAIEmbeddings(
    api_key=ALIBABA_API_KEY,
    base_url=ALIBABA_BASE_URL,
    model="text-embedding-v4",
    check_embedding_ctx_length=False#关键参数
)


zhipu_client = ZhipuAiClient(api_key= ZHIPU_API_KEY)  # 填写您自己的APIKey



# 方式1: 直接指定模型名并设置 cache_folder（如果模型已下载到本地，且结构符合 sentence-transformers 的预期）
gme_st = SentenceTransformer(
    LOCAL_GME_MODEL_PATH, # 或者使用本地路径 model_path
)

# 意图识别的小模型embedding
text_emb = CustomQwen3Embeddings(LOCAL_TEXT_EMB_PATH)