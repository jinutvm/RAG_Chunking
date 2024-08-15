from langchain_ai21 import AI21SemanticTextSplitter
from dotenv import load_dotenv
load_dotenv()

text = """### Retrieval Augmented Generation (RAG)

Language models (LLMs) are highly capable of providing information on a wide range of topics because they are trained on publicly available data. This extensive training enables their widespread use across various applications. However, one significant limitation of LLMs is their lack of access to proprietary information from companies or individuals, as they cannot access or learn from such restricted data.

To address this gap, Retrieval Augmented Generation (RAG) is employed. RAG enhances the response accuracy of LLMs by allowing them to pull information from specific documents, ensuring that the answers are based on precise and relevant sources.

### Retrieval Augmented Generation (RAG) Steps:

1. **Storing Documents in a Vector Store:**

   - Documents (such as PDFs, web pages, etc.) are divided into chunks, or smaller pieces of text.
   - These chunks are then embedded and stored in a vector store.

2. **Retrieving Relevant Text:**

   - The question is embedded using the same technique that was used to store the document chunks.
   - The question embedding is then searched within the vector store to find the relevant chunk embeddings that are closest to it.

3. **Generating Answers from Retrieved Context:**
   - The relevant chunks obtained from the vector store are provided to the LLM along with the question.
   - The LLM then answers the question based on the context provided by these chunks.

"""

semantic_text_splitter = AI21SemanticTextSplitter()
# chunks = semantic_text_splitter.split_text(text)  # split text into chunks
# print(f"The text has been split into {len(chunks)} chunks.")
# for chunk in chunks:
#     print(chunk)
#     print("====")
# documents = semantic_text_splitter.split_text_to_documents(text) # split text into documents
documents = semantic_text_splitter.create_documents(
    texts=[text], metadatas=[{"source": "from initial blog"}])
# split text into documents

print(documents)

"""
The text has been split into 5 chunks.
### Retrieval Augmented Generation (RAG)

Language models (LLMs) are highly capable of providing information on a wide range of topics because they are trained on publicly available data.

This extensive training enables their widespread use across various applications.

However, one significant limitation of LLMs is their lack of access to proprietary information from companies or individuals, as they cannot access or learn from such restricted data.

To address this gap, Retrieval Augmented Generation (RAG) is employed.

RAG enhances the response accuracy of LLMs by allowing them to pull information from specific documents, ensuring that the answers are based on precise and relevant sources.
====
### Retrieval Augmented Generation (RAG) Steps:

1.
====
**Storing Documents in a Vector Store:**

   - Documents (such as PDFs, web pages, etc.) are divided into chunks, or smaller pieces of text.

- These chunks are then embedded and stored in a vector store.
====
2. **Retrieving Relevant Text:**

   - The question is embedded using the same technique that was used to store the document chunks.

- The question embedding is then searched within the vector store to find the relevant chunk embeddings that are closest to it.
====
3. **Generating Answers from Retrieved Context:**
   - The relevant chunks obtained from the vector store are provided to the LLM along with the question.

- The LLM then answers the question based on the context provided by these chunks.
====

"""
