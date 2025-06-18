import os, argparse

# LangChain document loaders for PDFs and text files
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Tool to split long documents into chunks suitable for embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding model from OpenAI (used to convert text chunks into vectors)
from langchain_openai import OpenAIEmbeddings, OpenAI

# Vector database (FAISS) to store and search embeddings
from langchain_community.vectorstores import FAISS

# RetrievalQA chain combines document retrieval + LLM response
from langchain.chains import RetrievalQA


def load_documents(path: str):
    # Detect if the file is a PDF
    if path.lower().endswith(".pdf"):
        # Use PyPDFLoader to read and split the PDF into pages
        return PyPDFLoader(path).load()
    # Otherwise assume it's a plain .txt file
    return TextLoader(path, encoding="utf-8").load()


def build_vector_store(docs):
    # Split each document into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    # Convert the chunks into vector embeddings using OpenAI
    # and store them in a FAISS vector database
    return FAISS.from_documents(chunks, OpenAIEmbeddings())


def main(path, query):
    # Load and split the document into pages or text
    documents = load_documents(path)
    # Create a vector index from the document chunks
    vectordb = build_vector_store(documents)
    # Initialize a RetrievalQA chain:
    # - Uses OpenAI as the LLM
    # - Uses FAISS to retrieve relevant chunks
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )
    # Run the agent: ask it to summarize or answer the query using the docs
    answer = qa_chain.run(query)
    # Print the final answer to console
    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    # Handle CLI arguments
    parser = argparse.ArgumentParser(description="Basic Doc Q&A Agent")
    parser.add_argument("file", help="Path to PDF or .txt file")
    parser.add_argument("--query", default="Summarize key insights from this document")
    args = parser.parse_args()
    # Ensure the OpenAI key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("Please set OPENAI_API_KEY in your shell")
    # Run main logic
    main(args.file, args.query)