from langchain_text_splitters import RecursiveCharacterTextSplitter

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
# define the recursive text splitter with parameters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=10)

# use the text splitter to split the text into documents.
splitted_docs = text_splitter.create_documents([text])

print("Count of chunks", len(splitted_docs))
# Count of chunks 20

print("Index", "|", "Content of split", "|", "Length of split")
for idx, split in enumerate(splitted_docs):
    print(idx, " | ", split.page_content, " | ", len(split.page_content))

"""
Index | Content of split | Length of split
0  |  ### Retrieval Augmented Generation (RAG)  |  40
1  |  Language models (LLMs) are highly capable of providing information on a wide range of topics  |  92
2  |  of topics because they are trained on publicly available data. This extensive training enables  |  94
3  |  enables their widespread use across various applications. However, one significant limitation of  |  96
4  |  of LLMs is their lack of access to proprietary information from companies or individuals, as they  |  97
5  |  as they cannot access or learn from such restricted data.  |  57
6  |  To address this gap, Retrieval Augmented Generation (RAG) is employed. RAG enhances the response  |  96
7  |  response accuracy of LLMs by allowing them to pull information from specific documents, ensuring  |  96
"""
# Better than charactor split for generic text. See the first split. Only the heading is in the first split instead of trying to fit 100 charactors.
# similary check the line 5 from charactor and recursive. This provides more meaniful chunks
