''' hypothetical_document_embeddings 假设性文件嵌入
    1.先针对q,让llm给出一份与答案相近的假设性文档hyde_q替代原始q
    2.使用hyde_q从db中检索docs,提升向量检索的召回质量
    2.rag(q,docs)

            files --> docs --> chorma 
                                    |
                                    |                                       
question ---> llm ---> hyde ---> db_retriver ---> docs ---> llm ---> answer
'''

from llm.llm import custom_embd,llm
from utils.split_docs import get_texts_sub_docs
from utils.db import get_chroma_retriever
from utils.question_process import question_to_hyde
from utils.chat import rag_qa_chat

if __name__ == "__main__":

    # 1.文件加载切分[省略，已保存本地]
    # sub_docs = get_texts_sub_docs('files/agent.txt')
    # 2.构建Chroma数据
    retriever = get_chroma_retriever(custom_embd,directory='./temp/chorma_db/1')
    # 3.问题转化
    q = '对于大模型的agent来说，什么是任务分解?'
    hyde_q = question_to_hyde(q,llm)

    # 4.获取答案
    docs = retriever.invoke(hyde_q)
    context_str = "\n\n".join([d_i.page_content  for d_i in docs])
    answer = rag_qa_chat(q,context_str,llm)
    
    print(f"{answer}")





    

