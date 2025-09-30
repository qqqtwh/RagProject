''' 基本rag系统: 切分web数据 + chorma存储

files --> docs --> chorma
                    |
                    |
question --> db_retriver --> target_docs --> llm --> answer

'''
             
from llm.llm import custom_embd,llm
from utils.split_docs import get_texts_sub_docs
from utils.db import init_chroma_retriever
from utils.chat import rag_qa_chat

if __name__ == "__main__":
    
    # 1.文件加载切分
    sub_docs = get_texts_sub_docs('files/agent.txt')
    
    # 2.构建Chroma数据
    retriever = init_chroma_retriever(sub_docs,custom_embd,directory='./temp/chorma_db/1')
    
    # 3.从db中检索
    q = '什么是任务分解?'
    targets_docs = retriever.invoke(q) # [doc,doc,doc]
    targets_docs = '\n\n'.join([i.page_content for i in targets_docs])

    # 4.llm_qa
    response_str = rag_qa_chat(q,targets_docs,llm)
    print(response_str)
    
