import asyncio

from langchain_core.messages import ToolMessage

from graph.my_state import MultiModalRAGState
from milvus_db.db_retriever import m_re
from utils.embeddings_utils import local_gme_one
from utils.log_utils import log

# 自定义是为了替代：由langgraph框架自带的ToolNode(有大模型动态传参)
class SearchContextToolNode:
    """自定义类，来执行搜索的上下文工具"""

    def __init__(self,tools:list)->None:
        self.tools_by_name = {tool.name:tool for tool in tools}

    async def __call__(self,inputs:dict):
        if messages :=inputs.get('messages',[]):
            message = messages[-1]
        else:
            raise ValueError('No message found in input')
        outputs = []

        # 并行执行所有工具调用
        tasks =[]
        for tool_call in message.tool_calls:
            if tool_call.get('args') and 'query' in tool_call.get('args'):
                query = tool_call['args']['query']
                log.info(f'开始从上下文中检索:{query}')
            else:
                query = inputs.get('input_text')

            task = self.tools_by_name[tool_call['name']].ainvoke(
                {'query':query,'user_name':inputs.get('user')}
            )
            tasks.append((tool_call,task))
        # 等待所有异步调用完成
        tool_results = await asyncio.gather(*[task for _,task in tasks],return_exceptions=True)

        for (tool_call,_),tool_result in zip(tasks,tool_results):
            if isinstance(tool_result,Exception):
                # 错误处理
                tool_result = f'工具执行错误：:{str(tool_result)}'

            outputs.append(ToolMessage(
                content=tool_result,
                name=tool_call['name'],
                tool_call_id = tool_call['id']
            ))
        return {'messages':outputs}


def retriever_node(state:MultiModalRAGState):
    """
    检索知识库并返回
    :param state:
    :return:
    """
    if state.get("input_type") == "only_image":
        log.info(f"开始从知识库中检索图片：{state.get('input_image')}")
        # 构建图像输入数据
        input_data = [{'image':state.get('input_image')}]
        # 获取图片向量
        ok,embedding,status,retry_after= local_gme_one(input_data)
        results = m_re.dense_search(embedding,limit=3)
    else:
        # 构建文本输入数据
        input_data = [{'text':state.get('input_text')}]
        ok,embedding,status,retry_after= local_gme_one(input_data)
        results = m_re.hybrid_search(embedding,state.get('input_text'),limit=3)
    log.info(f"从知识库中检索到结果：{results}")

    # 返回文档内容
    images = [] # 图片路径
    docs = []
    print(results)
    for hit in results:
        # 过滤低分数
        distance = getattr(hit, 'distance', None)
        if distance is not None and isinstance(distance, (int, float)) and distance >= 0.7:
            if hit.get('category') =='image':
                images.append(hit.get('image_path'))
            docs.append({
                'text': hit.get('text', ''),
                'category': hit.get('category', ''),
                'image_path': hit.get('image_path', ''),
                "title": hit.get('title', ''),
                'file_name': hit.get('filename', ''),
            })
    # 返回并更改状态
    return {'context_retrieved':docs,'images_retrieved':images}
