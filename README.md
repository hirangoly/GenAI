# GenAI
**RAG-Based virtual assistant** (or chatbot) that answers questions only
from the content on the website, using Langchain.

**Overview**

This project implements a Retriever-Augmented Generation (RAG) system using LangChain to retrieve and summarize content from a given webpage. It utilizes OpenAI's LLM, FAISS vector database, and RecursiveCharacterTextSplitter for chunking text efficiently. The system is designed to:

* Load and extract text from a webpage.
* Split large documents into smaller chunks.
* Store vector representations in a FAISS index.
* Retrieve relevant document snippets based on a user query.
* Use an LLM to generate summaries and answers to user queries.

**Features**

✅ Extracts text from a given URL.
✅ Splits content into chunks for efficient processing.
✅ Stores and retrieves documents using FAISS.
✅ Uses OpenAI's LLM for summarization and Q&A.
✅ Fallbacks to LLM if no relevant documents are found.

**Installation**

1️⃣ Clone the Repository

git clone https://github.com/your-username/RAG-Webpage-Summarizer.git
cd RAG-Webpage-Summarizer

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Set Up OpenAI API Key

Create a .env file and add your OpenAI API key:

echo "OPENAI_API_KEY=your_openai_api_key" > .env

Or manually add it to .env:

OPENAI_API_KEY=your_openai_api_key

**Usage**

Run the Script

Modify the url variable inside main.py to your target webpage, then run:

python main.py

Query the Summarized Content

Modify the query variable to retrieve specific information:

query = "Summarize article about seven planets"

**Example Output:**

Generated Response:
 The BBC article discusses the discovery of seven exoplanets orbiting a distant star, highlighting their potential for habitability...

**Code Breakdown**

1️⃣ Load and Split Webpage Content

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

url = "https://www.bbc.com/news/technology"
loader = WebBaseLoader(url)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

2️⃣ Store in FAISS Vector Database

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_webpage_index")

3️⃣ Implement RAG-based Retrieval and Query Processing

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

4️⃣ Query the System

query = "Summarize article about seven planets"
response = qa_chain.run(query)
print("Generated Response:\n", response)

**Future Enhancements**

✅ Support multiple document sources.

✅ Improve response accuracy with better chunking strategies.

✅ Deploy as an API or chatbot.

✅ Use a different vector database (e.g., Pinecone, ChromaDB).


