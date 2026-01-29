from graph.my_state import MultiModalRAGState
from langgraph.graph.state import END


def route_only_image(state: MultiModalRAGState):
    """
    动态路由函数：用户输入是否只有图片类型
    :param state:
    :return:
    """
    if state.get('input_type') == 'only_image':
        return 'retriever_node'
    return 'first_chatbot'


def route_llm_or_retriever(state: MultiModalRAGState):
    """
    动态路由函数，如果上下文检索到结果，则进入可M节点，否则进入知识库检索节点
    """
    if messages := state.get("messages", []):
        tool_message = messages[-1]
    else:
        raise ValueError("No message found in input")
    if not tool_message.content or tool_message.content == "没有找到相关的历史上下文信息":
        return "retriever_node"
    return 'second_chatbot'


def route_retriever_evaluate(state: MultiModalRAGState):
    """
     动态路由函数，RAG是否搜索到相关文档
     """
    if docs := state.get("context_retrieved", []):
        return 'third_chatbot'
    else:
        return "fourth_chatbot"


def route_evaluate_node(state: MultiModalRAGState):
    """
    动态路由函数：如果用户仅仅输入图片，则不进行评估，（目前RAGAS还不支持多模态评估），其他情况下进入评估节点
    """
    if state.get('input_type') == 'only_image':
        return END
    return 'evaluate_node'


def route_human_node(state: MultiModalRAGState):
    """
    动态路由参数，如果评估后的分值低于0.6，则进入人工介入节点
    :param state:
    :return:
    """
    if state.get('evaluate_source') >= 0.8:
        return END
    return 'human_approval'


def route_human_approval_node(state: MultiModalRAGState):
    """
    动态路由函数，如果用户输入的是:approve则结束，否则进入网络搜索
    :param state:
    :return:
    """
    if state.get('human_answer') == 'approve':
        return END
    return 'fourth_chatbot'
