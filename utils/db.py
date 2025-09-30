from langchain_chroma import Chroma

# 构建Chroma数据
def init_chroma_retriever(sub_docs,embedding_model,directory='./temp/chorma_db/demo'):
    db = Chroma.from_documents(
        documents = sub_docs,
        embedding=embedding_model,
        persist_directory=directory,
        )
    
    retriever = db.as_retriever()

    return retriever

def get_chroma_retriever(embedding_model,directory='./temp/chorma_db/demo'):
    db = Chroma(
        embedding_function=embedding_model,
        persist_directory=directory,
        )
    
    retriever = db.as_retriever()

    return retriever