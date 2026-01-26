import os.path
from typing import List,Dict,Any

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from milvus_db.collections_ioerator import COLLECTION_NAME, client
from utils.embeddings_utils import image_to_base64, call_dashscope_once, local_gme_one


class MilvusRetriever:

    def __init__(self, collection_name: str, milvus_client: MilvusClient, top_k: int = 5):
        self.collection_name = collection_name
        self.milvus_client = milvus_client
        self.top_k = top_k

    def dense_search(self, query_embedding, limit=5):
        """
        密集向量检索
        :param query_embedding:
        :param limit:
        :return:
        """
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="dense",  # 密集向量中有图片和文本
            limit=limit,
            output_fields=["text", 'category', 'filename', 'image_path', 'title'],
            search_params=search_params,
        )
        return res[0]

    def sparse_search(self, query, limit=5):
        """
         稀疏向量搜索:全文检索
         :param query:  搜索的关键词
         :param limit:
         :return:
         """
        search_params = {"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}}
        res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field="sparse",  # 全文检索只能检索文本
            limit=limit,
            output_fields=["text", 'category', 'filename', 'image_path', 'title'],
            search_params=search_params,
        )
        return res[0]

    def hybrid_search(self, query_dense_embedding, query_sparse_embedding, sparse_weight=1.0, dense_weight=1.0,
                      limit=10):
        """
        混合搜索
        :param query_dense_embedding:
        :param query_sparse_embedding:
        :param sparse_weight:
        :param dense_weight:
        :param limit:
        :return:
        """
        dense_search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense", dense_search_params, limit=limit
        )

        sparse_search_params = {"metric_type": "BM25", 'params': {'drop_ratio_search': 0.2}}
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], "sparse", sparse_search_params, limit=limit
        )
        # 重排算法
        rerank = WeightedRanker(sparse_weight, dense_weight)
        return self.milvus_client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[sparse_req, dense_req],
            ranker=rerank,
            limit=limit,
            output_fields=['text', 'category', 'filename', 'image_path', 'title']
        )[0]

    def retrieve(self,query:str)->List[Dict[str,Any]]:
        """
        搜索
        :param query: 可能是图片（路径），也可能是文本
        :return:
        """
        if os.path.isfile(query):
            # 构建图像输入数据
            input_data = [{'image': image_to_base64(query)[0], 'factor': 1}]

            # 调用API获取图像嵌入向量
            ok, embedding, _, _ = local_gme_one(input_data)
        else:
            # 构建文本输入数据
            input_data = [{'text': query, 'factor': 1}]
            ok, embedding, _, _ = local_gme_one(input_data)

        results = []
        if ok:
            if os.path.isfile(query): # 纯图片不能进行混合检索
                results = self.dense_search(embedding,limit=self.top_k)
            else:
                results = self.hybrid_search(embedding,query,limit=self.top_k)

        docs = []
        print(results)
        for hit in results:
            docs.append({
                'text':hit.get('text',''),
                'category': hit.get('category',''),
                'im age_path': hit.get('image_path',''),
                "title": hit.get('title',''),
                'file_name': hit.get('filename',''),
            })
        return  docs

if __name__ == '__main__':
    m_re = MilvusRetriever(collection_name=COLLECTION_NAME,milvus_client=client)
    docs = m_re.retrieve("../output/images/06deb205fbda705dfc5a1d96fae0cdae.png")
    for doc in docs:
        print(doc)



