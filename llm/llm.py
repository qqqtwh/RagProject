from langchain_openai import ChatOpenAI
from openai import OpenAI
from typing import List
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('api_key')
base_url=os.getenv('base_url')
out_base_url=os.getenv('out_base_url')
model_name=os.getenv('model_name')
vision_model=os.getenv('vision_model')
multi_model=os.getenv('multi_model')
embedding_model=os.getenv('embedding_model')

# 禁用流式输出disable_streaming=True
llm = ChatOpenAI(base_url=base_url,api_key=api_key,model=model_name)
vlm = ChatOpenAI(base_url=base_url,api_key=api_key,model=vision_model)
mlm = ChatOpenAI(base_url=base_url,api_key=api_key,model=multi_model,streaming=True)
elm = ChatOpenAI(base_url=out_base_url,api_key=api_key,model=embedding_model)

class ParrotLinkEmbeddings(Embeddings):
    def __init__(self,model:str=embedding_model,batch_size=10,dim=128):
        self.model_name = model
        self.client = OpenAI(base_url=out_base_url,api_key=api_key)
        self.batch_size = batch_size
        self.dim=dim

    def _embed_batch(self,texts):
        res = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            dimensions=self.dim, # 向量维度 64, 128, 256, 512, 768, 1024
            encoding_format="float",
            
        )
        return [i.embedding for i in res.data] 

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []

        for i in range(0,len(texts),self.batch_size):
            batch = texts[i:i+self.batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:        
        return self.embed_documents([text])[0]

custom_embd = ParrotLinkEmbeddings(embedding_model)