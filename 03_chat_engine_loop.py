# Simple use of chat engine
# Same as 02 sample code but now running as a loop

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

load_dotenv()

llm = OpenAI(model="gpt-4o-mini")
data = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(data)

# Using chat mode = best; additional options explained below
chat_engine = index.as_chat_engine(chat_mode="best", llm=llm, verbose=True)
"""
Chat modes:
            - `ChatMode.BEST` (default): Chat engine that uses an agent (react or openai) with a query engine tool
            - `ChatMode.CONTEXT`: Chat engine that uses a retriever to get context
            - `ChatMode.CONDENSE_QUESTION`: Chat engine that condenses questions
            - `ChatMode.CONDENSE_PLUS_CONTEXT`: Chat engine that condenses questions and uses a retriever to get context
            - `ChatMode.SIMPLE`: Simple chat engine that uses the LLM directly
            - `ChatMode.REACT`: Chat engine that uses a react agent with a query engine tool
            - `ChatMode.OPENAI`: Chat engine that uses an openai agent with a query engine tool
"""

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = chat_engine.chat(text_input)
    print(f"Agent: {response}")
