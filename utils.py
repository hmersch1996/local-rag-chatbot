import chromadb
import llama_index
import ollama

def chroma_init(local_storage=None, collection_name=None):
    if local_storage == None:
        chroma_client = chromadb.EphemeralClient()
    
    else:
        chroma_client = chromadb.PersistentClient(path = local_storage)

    if collection_name == None:
        chroma_collection = chroma_client.get_or_create_collection("quickstart")

    else: 
        chroma_collection = chroma_client.get_or_create_collection(collection_name)

    return chroma_collection


from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

def create_llama_connectors(chroma_collection):

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    return storage_context

from llama_index.core import SimpleDirectoryReader



from typing import List

import pandas as pd
from llama_index.core.schema import TextNode


def data_to_df(nodes: List[TextNode]):
    """Convert a list of TextNode objects to a pandas DataFrame."""
    return pd.DataFrame([node.dict() for node in nodes])


from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser

def vsi(docs,storage_context, embed_model, chunk_size=None):

    if chunk_size == None:
        index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
        batch_size = 64
        )
    
    else: 
        sentence_splitter = SentenceSplitter(chunk_size=chunk_size)

        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
            transformations=[sentence_splitter],
            batch_size = 64
        )

    retriever = index.as_retriever(
    similarity_top_k=3,
    )

    return index, retriever

from llama_index.llms.ollama import Ollama

# llm = Ollama(model="mistral:latest", request_timeout=120.0)


def simple_rag(index, llm, consulta):
    query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    streaming=True,
    )

    response = query_engine.query(consulta)

    return response.print_response_stream()

from llama_index.core import PromptTemplate
def augmented_rag(index, llm, consulta, template):

    qa_template = PromptTemplate(template)


    query_engine = index.as_query_engine(
    llm=llm,
    similartiy_top_k=3,
    streaming=True,
    text_qa_template=qa_template,
    )

    response = query_engine.query(consulta)

    return response.print_response_stream()



from datetime import datetime
def calcular_duracion(tiempo_inicio, tiempo_fin, devolver=False):
    if isinstance(tiempo_inicio, (int,float)):
        tiempo_inicio = datetime.fromtimestamp(tiempo_inicio)
    if isinstance(tiempo_fin, (int,float)):
        tiempo_fin = datetime.fromtimestamp(tiempo_fin)

    duracion = tiempo_fin - tiempo_inicio

    total_segundos = int(duracion.total_seconds())

    horas = total_segundos // 3600

    minutos = (total_segundos%3600) // 60

    segundos = total_segundos % 60

    if devolver==True:
        return horas, minutos, segundos

    else:
        print(f"\nDuracion: {horas} horas, {minutos} minutos, {segundos} segundos\n")









