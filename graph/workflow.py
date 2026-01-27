from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.postgres import PostgresSaver

from graph.search_node import SearchContextToolNode
from graph.tools import search_context
from my_llm import mvltiModel_llm
from utils.log_utils import log
from graph.my_state import MultiModelRAGState, InvalidInputError

# 上下文检索工具列表
tools = [search_context]

# 工作流节点函数
def process_input(state:MultiModelRAGState,config:RunnableConfig):
    """处理用户输入"""
    user_name = config['configurable'].get('user_name','ZS')
    last_message = state['messages'][-1]
    log.info(f'用户 {user_name} 输入：{last_message}')
    input_type = 'has_text'
    text_content = None
    image_url = None
    # 检查输入的类型
    if isinstance(last_message,HumanMessage):
        if isinstance(last_message.content,list):
            content = last_message.content
            for item in content:
                # 提取纯文本内容
                if item.get('type') == 'text':
                    text_content = item.get('text',None)

                # 提取图片urk
                elif item.get('type') == 'image_url':
                    url = item.get('image_url',"").get('url')
                    if url: # 确保URL有效，是图片的base64格式的字符串
                        image_url = url
    else:
        raise InvalidInputError(f'用户输入的消息有误！ 原始输入：{last_message}')

    if not text_content and image_url:
        input_type = 'only_image'

    # 需要更新 的数据，和State一致
    return {
        "input_type": input_type,
        "text_content": text_content,
        "image_url": image_url,
        "user": user_name
    }

def first_chatbot(state:MultiModelRAGState,config:RunnableConfig):
    """
    第一次生成的回复或者决策（基于当前短期历史记录会话生成回复）
    """
    llm_with_tool = mvltiModel_llm.bind_tools([tools])
    system_message = SystemMessage(
        content=f"""
    你是文博行业的考古专家，如果输入给你的信息中包含相关内容，直接回答。
    否则不要自己直接回答
"""
    )

    message = llm_with_tool.invoke([*state['messages'],system_message])
    return {'messages',[message]}


store = InMemoryStore() # 短期记忆
DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
# 创建图
builder = StateGraph(MultiModelRAGState)

# 添加节点
builder.add_node('process_input',process_input) # 处理用户输入
builder.add_node('first_chatbot',first_chatbot)

search_context_node =  SearchContextToolNode(tools=tools)
builder.add_node('search_context',search_context_node)
builder.add_node('retriever_node',retriever_node)
builder.addnodee('second_chatbot',second_chatbot)
builder.add_node("third_chatbot", third_chatbot)
builder.add_node("evaluate_node", evaluate_answer)
builder.add_node("human_approval",human_approval)
builder.add_node("fourth_chatbot", fourth_chatbot)
builder.add_node("web_search_node", ToolNode(tools=web_tools))


with PostgresSaver.from_conn_string(DB_URI) as store:
    graph = builder.compile(store=store)

