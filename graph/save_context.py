import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from pymilvus import MilvusClient

from milvus_db.collections_ioerator import client
from utils.embeddings_utils import local_gme_one
from  utils.log_utils import log
from my_llm import  gme_st
from  typing import Dict,Any

# 全局线程池用户异步操作
thread_pool = ThreadPoolExecutor(max_workers = 5)

class OptimizedMilvusAsyncWriter:
    def __init__(self,client:MilvusClient,
                 collection_name:str="t_context_collection"):
        self.client = client
        self.collection_name = collection_name

    def _get_dense_vector(self,text:str):
        """异步生成稠密向量"""
        try:
            # 稠密向量生成（使用OpenAI或本地模型）
            ok, embedding, _, _  = local_gme_one([{'text': text, 'factor': 1}])
            return embedding
        except Exception as e:
            log.exception(f'向量生成失败：{e}')
            return None

    def _sync_insert(self,data:Dict[str,Any]):
        """同步插入到Milvus"""
        try:
            # 插入数据
            result = self.client.insert(collection_name=self.collection_name,data=data)
            log.info(f'[Milvus] 成功插入 {result['insert_count']} 条记录。IDS 示例: {result['ids'][:5]}')
        except Exception as e:
            log.exception(f'插入到数据到Milvus失败：{e}')

    async def async_insert(self,context_text:str,user:str,message_type:str):
        """异步插入数据"""
        data ={
            "context_text":context_text,
            "user": user,
            "timestamp": int(time.time()*1000),# 毫秒时间戳
            "message_type": message_type,
            "context_dense":self._get_dense_vector(context_text)
        }
        log.info(f"准备使用线程池异步插入数据：{data}")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(thread_pool,self._sync_insert,data)

def get_milvus_writer()->OptimizedMilvusAsyncWriter:
    return OptimizedMilvusAsyncWriter(client)