from langchain_core.tools import tool
from pymilvus import AnnSearchRequest, WeightedRanker

from milvus_db.collections_ioerator import CONTEXT_COLLECTION_NAME, client
from utils.embeddings_utils import local_gme_one
from utils.log_utils import log

@tool('search_context',parse_docstring=True)
def search_context(query: str=None, user_name: str=None)-> str:
    """
    根据用户的输入，检索与查询相关的长期历史上下文信息，然后给出正确的回答
    Args:
        query:(可选)用户刚刚输入的文本内容。
        user_name:(可选)当前的用户名。
    Returns:
        从历史上下文中检索到的结果。
    """
    # 构建文本输入数据
    input_data = [{'text':query}]
    # 获取嵌入向量
    ok,embedding,_,_ = local_gme_one(input_data)
    filter_expr = None
    if user_name:
        filter_expr = f'user == "{user_name}"' # 过滤搜索

    dense_search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    dense_req = AnnSearchRequest(
        [embedding], "dense", dense_search_params, limit=3,expr=filter_expr
    )

    sparse_search_params = {"metric_type": "BM25", 'params': {'drop_ratio_search': 0.2}}
    sparse_req = AnnSearchRequest(
        [query], "sparse", sparse_search_params, limit=3,expr=filter_expr
    )
    # 重排算法
    rerank = WeightedRanker(1.0, 1.0)
    res =  client.milvus_client.hybrid_search(
        collection_name=CONTEXT_COLLECTION_NAME,
        reqs=[sparse_req, dense_req],
        ranker=rerank,
        limit=3,
        output_fields=['text', 'category', 'filename', 'image_path', 'title']
    )[0]

    # 应用层过滤：只保留分数 >=min_score(0.55)的结果
    filtered_result = [item for item in res[0] if item.distance >=0.75]
    log.info(filtered_result)
    # 处理结果
    context_pieces= []
    for hit in filtered_result:
        context_pieces.append(f"{hit.get('context_text')}")

    return "\n".join(context_pieces) if context_pieces else "没有找到相关的历史上下文信息"