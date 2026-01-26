from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from numpy import ndarray
from sentence_transformers import SentenceTransformer

from utils.env_utils import ALIBABA_API_KEY, ALIBABA_BASE_URL


class CustomQwen3Embeddings(Embeddings):
    """自定义一个qwen3的Embedding和langchain整合的类"""

    def __init__(self, model_name):
        self.qwen3_embedding = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> ndarray:
        return self.qwen3_embedding.encode(texts)


mvltiModel_llm = ChatOpenAI(# 多模态大模型
    model='qwen-vl-plus',
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
