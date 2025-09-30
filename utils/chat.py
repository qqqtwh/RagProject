from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 基础QA回答
def rag_qa_chat(question,context_str,llm):
    rag_template = """你是问答任务的助理。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三句话，保持答案简洁。
问题: {question}
上下文: {context}
答案:"
"""
    
    prompt_format = ChatPromptTemplate.from_template(rag_template)
    format_messages = prompt_format.invoke({'question':question, 'context':context_str})
    response = llm.invoke(format_messages)
    final_answer = StrOutputParser().invoke(response)
    return final_answer

# 总结各个子问题的 qa 对获得最终answer
def summary_qa_chat(question,context_str,llm):
    rag_template = """以下是一组Q+a对：
{qa_content_list}
使用这些来综合问题的答案：
{question}
"""
    
    prompt_format = ChatPromptTemplate.from_template(rag_template)
    format_messages = prompt_format.invoke({'question':question, 'qa_content_list':context_str})
    response = llm.invoke(format_messages)
    final_answer = StrOutputParser().invoke(response)
    return final_answer


# 问题分为多个子问题后，结合上下文qa对回答当前问题
def qa_pairs_context_chat(question,q_a_pairs,context_str,llm):
    rag_template = """以下是你需要回答的问题：
---
{question}
---
以下是任何可用的背景问题+答案对：
---
{q_a_pairs}
---
以下是与该问题相关的其他背景：
---
{context}
---
使用上述上下文和任何背景问题+答案对来回答以下问题：
{question}
"""
    
    prompt_format = ChatPromptTemplate.from_template(rag_template)
    format_messages = prompt_format.invoke({'question':question, 'q_a_pairs':q_a_pairs, 'context':context_str})
    response = llm.invoke(format_messages)
    final_answer = StrOutputParser().invoke(response)
    return final_answer


# 综合 普通qa + 回退qa 获得最终回答
def step_back_context_chat(question,context_str,step_back_context_str,llm):
    rag_template = """你是一位世界知识专家。我将向你提出一个问题。你的回答应当全面，并且在以下上下文相关内容适用时，不得与之矛盾；如果上下文不相关，则可忽略。
{context}
{step_back_context}
原始问题：{question}
答案:
"""
    
    prompt_format = ChatPromptTemplate.from_template(rag_template)
    format_messages = prompt_format.invoke({'question':question, 'context':context_str, 'step_back_context':step_back_context_str})
    response = llm.invoke(format_messages)
    final_answer = StrOutputParser().invoke(response)
    return final_answer