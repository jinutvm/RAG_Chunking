from langchain_voyageai import VoyageAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import os
from dotenv import load_dotenv
load_dotenv()

# embeddings
embeddings = VoyageAIEmbeddings(
    voyage_api_key=os.getenv('VOYAGE_API_KEY'), model="voyage-2"
)

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

# text_splitter = SemanticChunker(
#     embeddings, breakpoint_threshold_type="percentile")

# docs = text_splitter.create_documents([text])
# print(len(docs))
# print(docs)

# text_splitter = SemanticChunker(
#     embeddings, breakpoint_threshold_type="standard_deviation")

# docs = text_splitter.create_documents([text])
# print(len(docs))
# print(docs)

text_splitter = SemanticChunker(
    embeddings, breakpoint_threshold_type="interquartile")

docs = text_splitter.create_documents([text])
print(len(docs))
print(docs)
