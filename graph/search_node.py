from langchain_core.messages import ToolMessage


# 自定义是为了替代：由langgraph框架自带的ToolNode(有大模型动态传参)
class SearchContextToolNode:
    """自定义类，来执行搜索的上下文工具"""

    def __init__(self,tools:list)->None:
        self.tools_by_name = {tool.name:tool for tool in tools}

    def __call__(self,inputs:dict):
        if messages :=inputs.get('messages',[]):
            message = messages[-1]
        else:
            raise ValueError('No message found in input')
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call['name']].invoke(
                {'query':inputs.get('input_text'),'user_name':inputs.get('user')}
            )
            outputs.append(ToolMessage(
                content=tool_result
            ))
        return {'messages',outputs}