
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 文件加载切分
def get_texts_sub_docs(file_path='files/agent.txt'):
    
    docs = TextLoader(file_path).load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," ",".",],
        chunk_size=1000,
        chunk_overlap=200,
    )
    sub_docs = text_splitter.split_documents(docs)

    return sub_docs