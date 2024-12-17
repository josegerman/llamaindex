import os
import dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

dotenv.load_dotenv()

documents = SimpleDirectoryReader("basic\data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("where are the bottles?")
print(response)