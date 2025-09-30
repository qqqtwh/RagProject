''' rank融合: 在 多版本提问提高回答质量 中, 将所有docs根据出现次数重新排序

            files --> docs --> chorma 
                                 |
                                 |
             |---> q1 |          |          |---> docs1 |
question --> |---> q2 |---> db_retriver --> |---> docs2 | --> docs_fusion -->llm --> answer
             |---> q3 |                     |---> docs3 |

'''

from llm.llm import custom_embd,llm
from langchain.load import dumps,loads
from utils.split_docs import get_texts_sub_docs
from utils.db import get_chroma_retriever
from utils.question_process import question_to_questions
from utils.chat import rag_qa_chat

if __name__ == "__main__":
    
    # 1.文件加载切分[省略，已保存本地]
    # sub_docs = get_texts_sub_docs('files/agent.txt')
    # 2.构建Chroma数据
    retriever = get_chroma_retriever(custom_embd,directory='./temp/chorma_db/1')
    # 3.问题转化
    q = '对于大模型的agent来说，什么是任务分解?'
    qs = question_to_questions(q,llm)


    # 4.从db中检索
    all_docs_list = [retriever.invoke(q_i) for q_i in qs] # [[doc,doc], [doc,doc]]

    # 5.根据出现次数计算每个doc得分
    fused_scores = {}
    for docs in all_docs_list:
        for index,doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1/(index+60)

    ranked = sorted(fused_scores.items(),key=lambda x:x[1], reverse=True)
    ranked_docs = [(loads(d_str),score) for d_str,score in ranked]
    context_str = "\n\n".join([i[0].page_content for i in ranked_docs])

    # 6.llm整合
    f =  rag_qa_chat(q,context_str,llm)
    print(f)

