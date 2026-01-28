from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper

from graph.my_state import MultiModalRAGState
from utils.log_utils import log
from my_llm import llm, text_emb
from evaluate.evaluate_self import RAGEvaluator
from langchain_core.messages import AIMessage

evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(text_emb)

#创建RAG评估器
rag_evaluator = RAGEvaluator(evaluator_llm, evaluator_embeddings)
async def evaluate_answer(state: MultiModalRAGState):
    """评估大模型的响应和用户输入之间的相关性"""
    context_retrieved = state.get('context_retrieved')
    input_text = state.get('input_text')
    last_message = state["messages"][-1]

    answer = None
    if isinstance(last_message, AIMessage):
        answer = last_message.content
    score = await rag_evaluator.evaluate_metrics(input_text, context_retrieved, answer)
    log.info(f"RAG Evaluation Score: {score}")
    return {"evaluate_source": float(score)}