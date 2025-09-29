from utils.llm import custom_elm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# 文档切分
raw_documents = TextLoader('1.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

q = 'What did the president say about Ketanji Brown Jackson'

# Chroma 向量库（持久化到本地）
chroma_db = Chroma(
    collection_name="user_memories",
    embedding_function=custom_elm,
    persist_directory="./chroma_db"  # 持久化路径
)

# 构建chorma数据库
db = chroma_db.from_documents(documents, custom_elm)

# 搜索1
docs1 = db.similarity_search(q)
print('-=-=\n',docs1)
# 搜索2
ev = custom_elm.embed_query(q)
docs2 = db.similarity_search_by_vector(ev)
print('-=-=\n',docs2)
