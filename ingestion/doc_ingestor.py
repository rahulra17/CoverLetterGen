from langchain.document_loaders import UnstructuredFileLoader
import os

def load_documents(folder_path: str):
    docs = []
    for file in os.listdir(folder_path):
        loader = UnstructuredFileLoader(os.path.join(folder_path, file))
        docs.extend(loader.load())
    return docs

def chunk_documents(docs, max_chars=3000):
    chunks = []
    buffer = ""
    try:
        for doc in docs:
            if len(buffer) + len(doc.page_content) <=max_chars:
                buffer += "\n\n" + doc.page_content
            else:
                chunks.append(buffer)
                buffer = doc.page_content
    except Exception as e:
        raise RuntimeError(f"Failed to chunk_documents: {e}")
    if buffer:
        chunks.append(buffer)
    else:
        print("Buffer was null, idk why but you should look into it")
    return chunks  


