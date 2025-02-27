# Data sources (documents): In this use case, we will have a single webpage

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

url = "https://www.bbc.com/news/technology"  # Replace with your target webpage
loader = WebBaseLoader(url)
documents = loader.load()

# Print extracted text
print(documents[0].page_content[:500])  # Preview first 500 characters

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

print(f"Total chunks: {len(docs)}")

# Summarizing Webpage Content

# from langchain.chat_models import ChatOpenAI
# from langchain.chains.summarize import load_summarize_chain

# llm = ChatOpenAI(model_name="gpt-4", temperature=0)
# summary_chain = load_summarize_chain(llm, chain_type="map_reduce")

# summary = summary_chain.run(docs)
# print(summary)

# Vector store: to store the document representation
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Save locally
vectorstore.save_local("faiss_webpage_index")


# query the stored webpage data
retriever = vectorstore.as_retriever()
# query = "What is the main topic of the webpage?"
# docs = retriever.get_relevant_documents(query)

# for doc in docs:
#     print(doc.page_content[:500])  # Show relevant snippet


from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set up the LLM (Language Model) for generative responses
llm = OpenAI(temperature=0.7)

# Set up the RetrievalQA chain (RAG system)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Query the system to get information about contraceptive pills from the retrieved documents
# query = "Share news about contraceptive pill"
query = "summarize article about seven planets"
# query = "What's the weather today" Note: real time doesn't give
# query = "capital of washington?"

# if not found in retrieved document, then search in LLM
if qa_chain:
	response = qa_chain.run(query)
else:
	response = llm(query)

# Display the result
print("Generated Response:\n", response)

