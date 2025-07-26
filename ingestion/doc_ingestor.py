from langchain.document_loaders import UnstructuredFileLoader
import os

def load_documents(folder_path: str):
    docs = []
    for file in os.listdir(folder_path):
        loader = UnstructuredFileLoader(os.path.join(folder_path, file))
        docs.extend(loader.load())
    return docs
