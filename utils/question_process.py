from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 将 q 转为多个不同表述但同语义的 qs
def question_to_questions(question,llm):
    template = """你是一个 AI 语言模型助手。你的任务是针对给定的用户问题，生成五个不同的表述版本，用于从向量数据库中检索相关文档。通过从多个角度重新表述用户问题，你的目标是帮助用户克服基于距离的相似性搜索所存在的一些局限性。请将这些改写后的问题用换行符分隔输出。原始问题：{question}"""
    prompt = ChatPromptTemplate.from_template(template)
    format_messages = prompt.invoke({'question':question})
    response = llm.invoke(format_messages)
    str_response = StrOutputParser().invoke(response)
    generated_queries = [q.strip() for q in str_response.split("\n") if q.strip()]
    return generated_queries


# 将问题 q 转为多个子问题 qs
def question_to_sub_questions(question,llm):
    template = """您是一个有用的助手，可以生成与输入问题相关的多个子问题。目标是将输入分解为一组可以单独回答的子问题。请将子问题用换行符分隔输出
生成与以下内容相关的多个搜索查询: {question}
输出（3个查询）:"""
    prompt = ChatPromptTemplate.from_template(template)
    format_messages = prompt.invoke({'question':question})
    response = llm.invoke(format_messages)
    str_response = StrOutputParser().invoke(response)
    generated_queries = [q.strip() for q in str_response.split("\n") if q.strip()]
    return generated_queries

# 复杂提问退化为通用提问
def step_back_question(question,llm):
    examples = [
        {
            "input": "警察可以进行合法逮捕吗？",
            "output": "警察可以做什么？",
        },
        {
            "input": "Jan Sindel 出生在哪个国家？",
            "output": "Jan Sindel 的个人经历是什么？",
        },
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一位世界知识专家。你的任务是“退一步”，将一个问题转述为一个更通用的“退一步问题”（step-back question），从而使其更容易回答。以下是几个示例：""",
            ),
            few_shot_prompt,
            ("human", "{question}"),
        ]
    )

    format_messages = prompt.invoke({'question':question})
    response = llm.invoke(format_messages)
    simple_question = StrOutputParser().invoke(response)

    return simple_question


# 将 q 转为 hyde
def question_to_hyde(question,llm):
    template = """请写一篇科学论文来回答这个问题。
问题: {question}
论文:
"""
    prompt = ChatPromptTemplate.from_template(template)
    format_messages = prompt.invoke({'question':question})
    response = llm.invoke(format_messages)
    str_response = StrOutputParser().invoke(response)
    return str_response