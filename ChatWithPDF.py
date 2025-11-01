import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


# ---------------------- #
# Configuration
# ---------------------- #
LOCAL_PDF_PATH = r"/Users/puravdoshi/Desktop/ChatWithPDF/sample.pdf"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
CHUNK_SIZE = 7500
CHUNK_OVERLAP = 100
COLLECTION_NAME = "local-rag"


# ---------------------- #
# Load the PDF
# ---------------------- #
def load_pdf(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    print(f"Loading PDF: {path}")
    loader = UnstructuredPDFLoader(file_path=path)
    data = loader.load()
    print(f"Loaded {len(data)} pages.")
    return data


# ---------------------- #
# Split into Chunks
# ---------------------- #
def split_into_chunks(documents):
    print("Splitting PDF into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")
    return chunks


# ---------------------- #
# Create Vector Database
# ---------------------- #
def create_vector_db(chunks):
    print("Generating embeddings and storing vectors...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL, show_progress=True),
        collection_name=COLLECTION_NAME,
    )
    print("Vector database created successfully.")
    return vector_db


# ---------------------- #
# Build RAG Chain
# ---------------------- #
def build_rag_chain(vector_db):
    print("Building retrieval-augmented generation chain...")

    llm = ChatOllama(model=LLM_MODEL)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are an AI assistant. Generate five different versions of the "
            "given question to improve document retrieval relevance.\n"
            "Original question: {question}"
        ),
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Retrieval chain built successfully.")
    return chain


# ---------------------- #
# Run Query
# ---------------------- #
def query_pdf(chain, user_query):
    print(f"\nQuery: {user_query}\n")
    response = chain.invoke(user_query)
    print("Response:\n")
    print(response)


# ---------------------- #
# 🚀 Main Function
# ---------------------- #
def main():
    try:
        documents = load_pdf(LOCAL_PDF_PATH)
        chunks = split_into_chunks(documents)
        vector_db = create_vector_db(chunks)
        chain = build_rag_chain(vector_db)

        # Example Query
        question = (
            "What is the most significant aspect of this PDF ?"
        )
        query_pdf(chain, question)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
