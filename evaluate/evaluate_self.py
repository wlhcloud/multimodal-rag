from typing import List, Dict

from ragas import SingleTurnSample
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics._context_precision import LLMContextPrecisionWithReference, LLMContextPrecisionWithoutReference
from ragas.metrics.collections import ContextRelevance

from milvus_db.collections_ioerator import COLLECTION_NAME, client
from milvus_db.db_retriever import MilvusRetriever
from my_llm import llm, text_emb


class RAGEvaluator:
    """
    RAG评估
    """

    def __init__(self, evaluator_llm, evaluator_embedding):
        self.evaluator_llm = evaluator_llm  # 推理模型
        self.evaluator_embedding = evaluator_embedding  # 嵌入模型

    @staticmethod
    def generate_answer(question: str, contexts: List[Dict]) -> str:
        """
        使用LLM基于检索到的上下文生成文本答案
        :param question:  用户的问题
        :param contexts:  检索到的上下文列表（包含text,category）等字段
        :return: 生成的中文答案
        """
        # 将检索到的上下文格式化为字符串，便于LLM理解
        # 每个上下文前加上“上下文x”标识，便于LLM区分
        context_str = "\n\n".join([f'上下文{i + 1}:{context['text']}' for i, context in enumerate(contexts)])

        # 提示词模板
        prompt = f"""
        你是一个AI助手，需要根据提供的上下文回答用户的问题。请确保你的回答基于提供的上下文，不要添加额外的信息
        用户问题:{question}
        检索到的上下文:
        {context_str}
        请基于以上上下文回答用户问题。
        """
        response = llm.invoke(prompt)
        return response.content

    async def evaluate_metrics(self, question: str, contexts: List[Dict], response: str, reference: str = None):
        """
        评估RAG模型
        :param question: 用户的问题
        :param contexts: 检索到的上下文
        :param response: LLM生成的答案
        :param reference: 参考答案（用户评估的基准答案，通常为已知的正常答案）
        """
        # 创建评估样本（SingleTurnSample）
        sample = SingleTurnSample(
            user_input=question,  # 用户问题
            retrieved_contexts=[context['text'] for context in contexts],  # 检索到的上下文
            response=response,  # AI生成的响应
            reference=reference,  # 参考答案
        )

        # 初始化评估指标
        if reference:
            # 如果有参考答案，则初始化指标为LLMContextPrecisionWithReference
            context_precision = LLMContextPrecisionWithReference(llm=self.evaluator_llm)
        else:
            # 如果没有参考答案，则初始化指标为LLMContextPrecisionWithoutReference
            context_precision = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)

        # 执行评估指标，得到结果
        score = await context_precision.single_turn_ascore(sample)
        print(f"上下文精度评分: {score:.3f}")
        return score

    async def evaluate_context(self, question: str, contexts: List[str])->float:
        """上下文相关性评估：检索到的上下文（快或段落）是否与用户输入相关"""
        # 0 -> 完全不相关； 1->部分相关；2->完全相关
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts
        )
        scorer= ContextRelevance(llm=self.evaluator_llm)
        metric_result = await scorer.ascore(question,contexts)
        return metric_result.value


evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embedding = LangchainEmbeddingsWrapper(text_emb)

# 创建评估器
rag_evaluator = RAGEvaluator(evaluator_llm=evaluator_llm, evaluator_embedding=evaluator_embedding)

async def main():
    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embedding = LangchainEmbeddingsWrapper(text_emb)

    # 创建评估器
    rag_evaluator = RAGEvaluator(evaluator_llm=evaluator_llm, evaluator_embedding=evaluator_embedding)

    question = "琉璃珠是谁提供的？"
    # 检索上下文（从Milvus数据库中获取）
    m_re = MilvusRetriever(COLLECTION_NAME, client)
    contexts = m_re.retrieve(question)

    generated_answer = rag_evaluator.generate_answer(question,contexts)
    print(f"生成的答案:{generated_answer}")

    await  rag_evaluator.evaluate_metrics(question,contexts,generated_answer)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())