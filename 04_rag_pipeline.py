import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader

load_dotenv()

# 1. Load data
documents = SimpleDirectoryReader("data").load_data()


# 2. Create index


# 3. Create vector store index with Chroma


# 4. Create query engine