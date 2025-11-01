# 📄 Local PDF RAG Chatbot

A Python script implementing a Retrieval-Augmented Generation (RAG) system to chat with a local PDF document. It leverages **LangChain** for orchestration, **Ollama** for running open-source embeddings and a Large Language Model (LLM), and **Chroma** as the vector store.

---

## ✨ Features

* **PDF Loading:** Uses `UnstructuredPDFLoader` to ingest documents.
* **Text Chunking:** Employs `RecursiveCharacterTextSplitter` for optimal text segmentation.
* **Local Embeddings & LLM:** Utilizes **Ollama** to host and use the `nomic-embed-text` embedding model and the `llama3` chat model locally, ensuring privacy and speed.
* **Vector Store:** Uses **Chroma** to store and retrieve document embeddings.
* **Multi-Query Retrieval:** Implements `MultiQueryRetriever` to generate multiple perspectives on the user's question, improving the relevance of retrieved context.

---

## 🛠️ Prerequisites

Before running the script, ensure you have the following installed:

1.  **Ollama:** Must be installed and running on your system.
    * [Ollama Installation Guide](https://ollama.com/)
2.  **Ollama Models:** Pull the required models using the Ollama CLI:
    ```bash
    ollama pull nomic-embed-text
    ollama pull llama3
    ```
3.  **Python Environment:** Set up a virtual environment and install the necessary Python packages.

---

## 💻 Installation and Setup

1.  **Clone the repository (if applicable) or save the code:**
2.  **Install Python dependencies:**

    ```bash
    pip install langchain-community langchain-text-splitters langchain-core unstructured pydantic chromadb
    ```
    *Note: `langchain` and its integrations are installed via the above packages.*

3.  **Place your PDF:**
    * Place the PDF you want to query in a designated location.
    * **Update the `LOCAL_PDF_PATH`** variable in the Python script (`main.py` or similar) to point to your PDF file.

    ```python
    LOCAL_PDF_PATH = r"/Users/puravdoshi/Desktop/ChatWithPDF/sample.pdf" # <-- Update this path
    ```

---

## 🚀 Usage

Run the script from your terminal:

```bash
python your_script_name.py
