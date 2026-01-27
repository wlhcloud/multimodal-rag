from typing import Optional,Dict,Literal,List

from langgraph.graph import MessagesState


class MultiModelRAGState(MessagesState):
    """
    状态数据结构
    """

    input_type: Literal['has_text','only_image'] # 用户输入类型
    context_retrieved: Optional[List[Dict[str,str]]] # 从向量数据库中检索到的文本内容
    images_retrieved: Optional[List[str]] # 从向量数据库中检索的图片路径
    needs_retrieval: Optional[bool] # 是否需要检索
    evaluate_source: Optional[float] # 评估分数
    final_response: Optional[str] # 最终回答

    input_image: Optional[str] # 用户输入的图片，里面是base64编码的图片
    input_text: Optional[str] # 用户输入的文本
    user: str

# 自定义异常类
class InvalidInputError(Exception):
    """自定义异常，用于表示无效输入"""
    def __init__ (self, message: str, error_code: int = 400):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)