import asyncio
import os.path
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import AIMessage

from graph.all_router import route_evaluate_node, route_human_node, route_human_approval_node, route_only_image, \
    route_llm_or_retriever, route_retriever_evaluate
from graph.evaluate_node import evaluate_answer
from graph.save_context import get_milvus_writer
from graph.search_node import SearchContextToolNode, retriever_node
from graph.tools import search_context, network_search
from my_llm import multiModal_llm
from utils.common_utils import draw_graph
from utils.embeddings_utils import image_to_base64
from utils.log_utils import log
from graph.my_state import MultiModalRAGState, InvalidInputError
from langgraph.graph.state import END, START

# 上下文检索工具列表
tools = [search_context]
web_tools = [network_search]


# 工作流节点函数
def process_input(state: MultiModalRAGState, config: RunnableConfig):
    """处理用户输入"""
    user_name = config['configurable'].get('user_name', 'ZS')
    last_message = state['messages'][-1]
    log.info(f'用户 {user_name} 输入：{last_message}')
    input_type = 'has_text'
    text_content = None
    image_url = None
    # 检查输入的类型
    if isinstance(last_message, HumanMessage):
        if isinstance(last_message.content, list):
            content = last_message.content
            for item in content:
                # 提取纯文本内容
                if item.get('type') == 'text':
                    text_content = item.get('text', None)

                # 提取图片urk
                elif item.get('type') == 'image_url':
                    url = item.get('image_url', "").get('url')
                    if url:  # 确保URL有效，是图片的base64格式的字符串
                        image_url = url
    else:
        raise InvalidInputError(f'用户输入的消息有误！ 原始输入：{last_message}')

    if not text_content and image_url:
        input_type = 'only_image'

    # 需要更新 的数据，和State一致
    return {
        "input_type": input_type,
        "input_text": text_content,
        "image_url": image_url,
        "user": user_name
    }


def first_chatbot(state: MultiModalRAGState):
    """
    第一次生成的回复或者决策（基于当前短期历史记录会话生成回复）
    """
    llm_with_tool = multiModal_llm.bind_tools(tools)
    system_message = SystemMessage(
        content=f"""
    你是文博行业的考古专家，如果输入给你的信息中包含相关内容，直接回答。
    否则不要自己直接回答
"""
    )

    message = llm_with_tool.invoke([*state['messages'], system_message])
    return {'messages': [message]}


def second_chatbot(state: MultiModalRAGState):
    """
    第二次生產回復（基于检索历史上下文生成回复，检索到的历史上下文在ToolMessage里面
    :param state:
    :return:
    """
    return {"messages": [multiModal_llm.invoke(state['messages'])]}


def third_chatbot(state: MultiModalRAGState):
    """处理多模态请求并返回Markdown格式的结果"""
    context_retrieved = state.get('context_retrieved')
    images = state.get('images_retrieved')

    # 处理上下文列表
    count = 0
    context_pieces = []
    for hit in context_retrieved:
        count += 1
        context_pieces.append(f"检索后的内容{count}：\n {hit.get('text')} \n 资料来源：{hit.get('filename')}")

    context = "\n\n".join(context_pieces) if context_pieces else "没有检索到相关的上下文信息"
    input_text = state.get('input_text')
    input_image = state.get('input_image')

    # 构建系统提示词
    system_prompt = f"""
    请根据用户输入和以下检索到的上下文内容生成响应，如果上下文内容中没有相关答案，请直接说明，不要自己真接输出答案。
    要求:
    1.响应必须使用Markdown格式
    2.在响应文字下方显示所有相关图片，图片的路径列表为{images}，使用Markdown图片语法:
    3.在相关图片下面的最后一行显示上下文引用来源(来源文件名)
    4.如果用户还输入了图片，请也结合上下文内容，生成文本响应内容。
    5.如果用户还输入了文本，请结合上下文内容，生成文本响应内容。
    
    上下文内容：
    {context}
    """

    # 构建用户消息内容
    user_content = []
    if input_text:
        user_content.append({"type": "text", "text": input_text})
    if input_image:
        user_content.append({"type": "image_url", "image_url": {"url": input_image}})
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_content)
    ]
    )
    chain = prompt | multiModal_llm

    llm_result = chain.invoke({})
    return {"messages": [llm_result]}


def human_approval(state: MultiModalRAGState):
    log.info('已经进入了人工审批节点')
    log.info(f'当前的状态中的人工审批信息:{state["human_answer"]}')


def fourth_chatbot(state: MultiModalRAGState):
    """网络搜索工具绑定的大模型，第四大模型调用"""
    llm_tools = multiModal_llm.bind_tools(web_tools)
    input_text = state.get('input_text')

    system_message = SystemMessage(
        content='你是一个智能体助手，请根据用户输入和互联网搜索结果，给出合理的回答。')
    message = HumanMessage(
        content=[
            {'type': 'text', 'text': input_text}
        ]
    )
    return {'messages': [llm_tools.invoke([system_message, state.get('messages')[-1], message])]}


# 创建图
builder = StateGraph(MultiModalRAGState)

# 添加节点
builder.add_node('process_input', process_input)  # 处理用户输入
builder.add_node('first_chatbot', first_chatbot)

search_context_node = SearchContextToolNode(tools=tools)
builder.add_node('search_context', search_context_node)
builder.add_node('retriever_node', retriever_node)
builder.add_node('second_chatbot', second_chatbot)
builder.add_node("third_chatbot", third_chatbot)
builder.add_node("evaluate_node", evaluate_answer)
builder.add_node("human_approval", human_approval)
builder.add_node("fourth_chatbot", fourth_chatbot)
builder.add_node("web_search_node", ToolNode(tools=web_tools))

# 添加边
builder.add_edge(START, "process_input")
builder.add_conditional_edges("process_input", route_only_image, {
    "retriever_node": "retriever_node",
    "first_chatbot": "first_chatbot"
})

builder.add_conditional_edges("first_chatbot", tools_condition, {
    "tools": "search_context",
    END: END
})
builder.add_conditional_edges("search_context", route_llm_or_retriever, {
    "retriever_node": "retriever_node",
    "second_chatbot": "second_chatbot"
})

# builder.add_conditional_edges("retriever_node", route_retriever_evaluate, {
#     "third_chatbot": "third_chatbot",
#     "fourth_chatbot": "fourth_chatbot"
# })
builder.add_edge('retriever_node', 'third_chatbot')  # 所有的结果都需要进行评估

# builder.add_edge('second_chatbot','evaluate_node') # 所有的结果都需要进行评估

builder.add_conditional_edges("third_chatbot", route_evaluate_node, {
    "evaluate_node": "evaluate_node",
    END: END
})
builder.add_conditional_edges("evaluate_node", route_human_node, {
    "human_approval": "human_approval",
    END: END
})
builder.add_conditional_edges("human_approval", route_human_approval_node, {
    "fourth_chatbot": "fourth_chatbot",
    END: END
})
builder.add_conditional_edges("fourth_chatbot", tools_condition, {'tools': 'web_search_node', END: END})
builder.add_edge('web_search_node', 'fourth_chatbot')

checkpointer = InMemorySaver()
DB_URI = "postgresql://postgres:postgres@www.wlhcloud.top:9123/postgres?sslmode=disable"
pg_store = None
with PostgresSaver.from_conn_string(DB_URI) as store:
    post_pg_saver = pg_store

graph = builder.compile(
    checkpointer=checkpointer,
    store=pg_store,
    interrupt_before=['human_approval']  # 添加中断节点 静态的人工介入
)
# draw_graph(graph, 'graph.rag_png')

session_id = str(uuid.uuid4())

# 配置参数，包含用户ID和线程ID
config = {
    'configurable': {
        "user_name": "WLH",
        # 检查点由session_id访问
        "thread_id": session_id
    }
}


def update_state(user_answer, config):
    """让人工介入"""
    if user_answer == "approve":
        new_message = "approve"
    else:
        new_message = "rejected"
    # 把人为输入的，存入图的state中
    graph.update_state(
        config=config,
        values={"human_answer": new_message}
    )


def pretty_print_messages(chunk, last_message):
    chunk["messages"][-1].pretty_print()


async def execute_graph(user_input: str) -> str:
    """执行工作流的函数"""
    result = ""  # AI助手的最后一条消息
    current_state = graph.get_state(config)
    log.info(f"【初始状态】current_state.next: {current_state.next}")
    log.info(f"【初始状态】current_state.values: {current_state.values.keys()}")
    if current_state.next:  # 出现了工作流中断
        # 通过提供关于请求的更改/改变主意的指示来满足图的继续执行
        update_state(user_input, config)
        # 恢复工作流
        async for chunk in graph.astream(None, config, stream_mode="values"):
            pretty_print_messages(chunk, last_message=True)

    else:
        image_base64 = None
        text = None
        if '&' in user_input:
            text = user_input.split('&')[0]
            image = user_input.split('&')[1]
            if image and os.path.isfile(image):
                image_base64 = {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64(image)[0]}
                }
        elif os.path.isfile(user_input):
            image_base64 = {
                "type": "image_url",
                "image_url": {"url": image_to_base64(user_input)[0]}
            }
        else:
            text = user_input

        message = HumanMessage(
            content=[
            ]
        )
        if text:
            message.content.append({"type": "text", "text": text})
        if image_base64:
            message.content.append(image_base64)
        async for chunk in graph.astream({'messages': [message]}, config, stream_mode='values'):
            pretty_print_messages(chunk, last_message=True)

    current_state = graph.get_state(config)
    log.info(f"【执行后状态】current_state.next: {current_state.next}")
    log.info(f"【执行后状态】是否中断: {bool(current_state.next)}")
    if current_state.next:  # 出现了工作流的中断
        output = ('由于系统自我评估后，发现AI的回复不是非常准确，您是否 认可以下输出？\n'
                  '如果认可,请输入：approve ，否则请输入 rejected ,系统将会重新生成回复 ！')
        result = output
        log.warning(f"【分支走向】进入人工审批分支，next={current_state.next}")
    else:
        # 异步写入到AI响应到Milvus(使用缓冲区优化）
        log.info(f"【分支走向】进入Milvus写入分支！")
        mess = current_state.values.get('messages', [])
        log.info(f"【消息列表】messages长度: {len(mess)}，最后一条类型: {type(mess[-1]) if mess else '空'}")

        if mess:
            if isinstance(mess[-1], AIMessage):
                log.info(f'开始写入Milvus：')
                task = asyncio.create_task(
                    get_milvus_writer().async_insert(
                        context_text=mess[-1].content,
                        user=current_state.values.get('user', "admin"),
                        message_type="AIMessage"
                    )
                )
                await task
    return result


async def main():
    # 执行工作流
    while True:
        user_input = input('用户输入（文本和图片用&隔开）')
        if user_input.lower() in ['exit', 'quit', '退出']:
            break
        res = await execute_graph(user_input)
        if res:
            print('AI：', res)


if __name__ == '__main__':
    asyncio.run(main())
