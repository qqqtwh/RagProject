''' 问题q拆解为多个子问题qi, qi + 历史qa对递归回答

            files --> docs --> chorma 
                                 |
                                 |
             |---> sub_q1 ---> db_retriver ---> docs1 ---> llm ---> answer1(+sub_q1)
                                                                        ↓
question --> |---> sub_q2 ---> db_retriver --->             docs2 ---> llm ---> answer2(+sub_q2)
                                                                                   ↓
             |---> sub_q3 ---> db_retriver --->                        docs3 ---> llm ---> answer

'''

from llm.llm import custom_embd,llm
from utils.split_docs import get_texts_sub_docs
from utils.db import get_chroma_retriever
from utils.question_process import question_to_sub_questions
from utils.chat import rag_qa_chat,qa_pairs_context_chat

if __name__ == "__main__":

    # 1.文件加载切分[省略，已保存本地]
    # sub_docs = get_texts_sub_docs('files/agent.txt')
    # 2.构建Chroma数据
    retriever = get_chroma_retriever(custom_embd,directory='./temp/chorma_db/1')
    # 3.问题转化
    q = '对于大模型的agent来说，什么是任务分解?'
    qs = question_to_sub_questions(q,llm)

    # 4.子问题结合历史回答递归回答
    q_a_pairs = ""
    for index, q_i in enumerate(qs):
        # 通过基础rag获得当前子问题的答案
        docs_i = retriever.invoke(q_i)
        context_str_i = "\n\n".join([d_i.page_content  for d_i in docs_i])
        answer_i = rag_qa_chat(q,context_str_i,llm)
        # 构建问答对
        q_a_pairs += f"问题: {q_i}\n回答: {answer_i}\n\n".strip()
        
        if index>0:
            answer_i = qa_pairs_context_chat(q_i,q_a_pairs,context_str_i,llm)
    
    print(answer_i)








    

