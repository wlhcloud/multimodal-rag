from langchain_community.tools.asknews.tool import SearchInput
from langchain_core.tools import tool
from pymilvus import AnnSearchRequest, WeightedRanker

from milvus_db.collections_ioerator import CONTEXT_COLLECTION_NAME, client
from my_llm import zhipu_client
from utils.embeddings_utils import local_gme_one
from utils.log_utils import log

@tool('search_context',parse_docstring=True)
def search_context(query: str, user_name: str=None)-> str:
    """根据用户的输入，检索与查询相关的长期历史上下文信息，然后给出正确的回答

    Args:
        query: 用户刚刚输入的文本内容。
        user_name: 当前的用户名（可选）。

    Returns:
        str: 从历史上下文中检索到的结果。
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
        [embedding], "context_dense", dense_search_params, limit=3,expr=filter_expr
    )

    sparse_search_params = {"metric_type": "BM25", 'params': {'drop_ratio_search': 0.2}}
    sparse_req = AnnSearchRequest(
        [query], "context_sparse", sparse_search_params, limit=3,expr=filter_expr
    )
    # 重排算法
    rerank = WeightedRanker(1.0, 1.0)
    res =  client.hybrid_search(
        collection_name=CONTEXT_COLLECTION_NAME,
        reqs=[sparse_req, dense_req],
        ranker=rerank,
        limit=3,
        output_fields=['context_text']
    )[0]

    # 应用层过滤：只保留分数 >=min_score(0.55)的结果
    filtered_result = [item for item in res if item.distance >=0.75]
    log.info(f'上下文中检索：{filtered_result}')
    # 处理结果
    context_pieces= []
    for hit in filtered_result:
        context_pieces.append(f"{hit.get('context_text')}")

    return "\n".join (context_pieces) if context_pieces else "没有找到相关的历史上下文信息"



@tool("my_search_tool", args_schema=SearchInput, description="专门搜索互联网中的内容")
def my_search(query: str) -> str:
    """
    专门搜索互联网中的内容
    :param query:
    :return str:
    """
    print(f"在查询{query}。。。。。")
    try:
        response = zhipu_client.web_search.web_search(
            search_engine="search_pro",
            search_query=query,
            count=15,  # 返回结果的条数，范围1-50，默认10
            search_domain_filter="www.sohu.com",  # 只访问指定域名的内容
            search_recency_filter="noLimit",  # 搜索指定日期范围内的内容
            content_size="high"  # 控制网页摘要的字数，默认medium
        )
        print(response)
        if response.search_result:
            return "\n\n".join([d.content for d in response.search_result])
    except Exception as e:
        print(e)
        return "没有查询到任何内容"
