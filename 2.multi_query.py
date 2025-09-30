''' 多版本提问提高回答质量: 将question转为多个不同版本转述，分别从数据库中获取答案，最后llm整合输出

            files --> docs --> chorma 
                                 |
                                 |
             |---> q1 |          |          |---> docs1 |
question --> |---> q2 |---> db_retriver --> |---> docs2 | --> llm --> answer
             |---> q3 |                     |---> docs3 |

'''

from llm.llm import custom_embd,llm
from utils.split_docs import get_texts_sub_docs
from utils.db import get_chroma_retriever
from utils.question_process import question_to_questions
from utils.chat import rag_qa_chat
from langchain.load import dumps,loads


if __name__ == "__main__":
    
    # 1.文件加载切分[省略，已保存本地]
    # sub_docs = get_texts_sub_docs('files/agent.txt')

    # 2.构建Chroma数据索引
    retriever = get_chroma_retriever(custom_embd,directory='./temp/chorma_db/1')

    # 3.问题转化
    q = '对于大模型的agent来说，什么是任务分解?'
    qs = question_to_questions(q,llm)

    # 4.从db中检索并去重
    all_docs = [doc for q_i in qs for doc in retriever.invoke(q_i) ] # [doc,doc,doc]
    # umps(doc) 转为唯一字符串表示  --> set() 去重 --> loads(s)还原为 Document 对象
    unique_docs = [loads(doc_str) for doc_str in set([dumps(doc) for doc in all_docs])]
    context_str = "\n\n".join([d.page_content for d in unique_docs])

    # 5.llm整合问答
    f =  rag_qa_chat(q,context_str,llm)
    print(f)
    



    
