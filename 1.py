from utils.llm import llm
import asyncio
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

if __name__ == "__main__":
    
    # 
    

    web_path = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    web_loader = WebBaseLoader(
        web_paths=[web_path],
        bs_kwargs={"parse_only":bs4.SoupStrainer(class_=["post-content", "post-title", "post-header"])},
    )

    docs = web_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," ",".",],
        chunk_size=1000,
        chunk_overlap=200,
    )

    splits = text_splitter.split_documents(docs)

    for i in splits:
        print()
        print(i)