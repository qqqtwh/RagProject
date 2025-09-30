''' 将question回退为通用back_question，综合考虑两者answer

    files --> docs --> chorma 
                            |
                            |                                       
question        ---> db_retriver ---> docs ---> llm ---> answer --------->  llm ---> answer
    ↓                                                                ↑
   llm                                                               |
    ↓                                                                |                                                                                                                                                                                                           ↓
back_question   ---> db_retriver ---> back_docs ---> llm ---> back_answer

'''

from llm.llm import custom_embd,llm
from utils.split_docs import get_texts_sub_docs
from utils.db import get_chroma_retriever
from utils.question_process import step_back_question
from utils.chat import rag_qa_chat,step_back_context_chat

if __name__ == "__main__":

    # 1.文件加载切分[省略，已保存本地]
    # sub_docs = get_texts_sub_docs('files/agent.txt')
    # 2.构建Chroma数据
    retriever = get_chroma_retriever(custom_embd,directory='./temp/chorma_db/1')
    # 3.问题转化
    q = '对于大模型的agent来说，什么是任务分解?'
    step_back_q = step_back_question(q,llm)

    # 4.获取答案

    # 原始问题对应答案
    docs = retriever.invoke(q)
    context_str = "\n\n".join([d_i.page_content  for d_i in docs])
    answer = rag_qa_chat(q,context_str,llm)
    # 退化问题对应答案
    step_back_docs = retriever.invoke(step_back_q)
    step_back_context_str = "\n\n".join([d_i.page_content  for d_i in step_back_docs])
    step_back_answer = rag_qa_chat(step_back_q,step_back_context_str,llm)
    
    # 5.最终答案
    final_answer = step_back_context_chat(q,answer,step_back_answer,llm)
    print(final_answer)

    







    

