import chromadb
import uuid
import os
from chromadb.config import Settings
import torch
from typing import List, Optional, Dict
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
import requests
from io import BytesIO
import shutil
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=True
)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

embedder = SentenceTransformer('intfloat/multilingual-e5-large')
print("Model loaded!")


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    # ignore_mismatched_sizes=True,
    low_cpu_mem_usage=False
)
print("Model loaded!")

llm_pipeline = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    num_beams=1,
    top_p=0.95,
    repetition_penalty=1.1,
    return_full_text=False
)

class ChromaRAG:    
    def __init__(self, persist_directory='./chroma_db'):
        self.persist_directory = persist_directory

        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        self.collection_name = 'documents'
        self.collection = self.get_or_create_collection_db()
        self.documet_registry = {}

        self.response_cache = {}
        self.cache_path = Path('./cache/responses.json')
        self.cache_path.parent.mkdir(exist_ok=True)

        self.chunks = []
        self.embeddings = None
        self.embeddings_cache = {}
        self.conversation_history = []

        print(f'Documents in DB {self.collection.count()}')

    def get_or_create_collection_db(self):
        try:
            collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=None
            )
            return collection

        except:
            collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={'hnsw:space': 'cosine'}
            )
            print('Create new collection')
            return collection
    
    def add_documet(self, source: str, source_type: str = 'url', metadata: Optional[Dict] = None) -> str:
        doc_id = str(uuid.uuid4())
        
        if source_type == 'url':
            response = requests.get(source)
            pdf_file = BytesIO(response.content)
            doc_name = source.split('/')[-1]
        else:
            pdf_file = source
            doc_name = os.path.basename(source)
        

        reader = PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'

        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")

        print("Chunking text...")
        self.chunks = self._chunk_text(text, chunk_size=800, overlap=100)
        print(f"Created {len(self.chunks)} chunks")
        

        print("Generating embeddings...")
        self.embeddings = embedder.encode(self.chunks, show_progress_bar=True)
        print("Ready to answer commands!")

        base_metadata = {
            'document_id': doc_id,
            'document_name': doc_name,
            'source': source,
            'added_date': datetime.now().isoformat(),
            'chunk_count': len(self.chunks)
        }

        if metadata:
            base_metadata.update(metadata)
    
        # added chunk_ids and metadatas in ChromaDB
        chunk_ids = [f'{doc_id}_chunk_{i}' for i in range(len(self.chunks))]
        metadatas = [{**base_metadata, 'chunk_index': i} for i in range(len(self.chunks))]
            
        self.collection.add(
            embeddings=self.embeddings.tolist(),
            documents=self.chunks,
            ids=chunk_ids,
            metadatas=metadatas
        )

        self.document_registry[doc_id] = {
            "name": doc_name,
            "chunks": len(self.chunks),
            "source": source,
            "added_date": base_metadata["added_date"]
        }
        print(f'Document {doc_name} added')
        return doc_id
    
    def list_document(self) -> List[Dict]:
        results = self.collection.get(include=['metadatas'])

        documents = {}
        for metadata in results['metadatas']:
            doc_id = metadata.get('document_id')
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "id": doc_id[:8] + "...",
                    "name": metadata.get('document_name', 'Unknown'),
                    "source": metadata.get('source', 'Unknown'),
                    "added_date": metadata.get('added_date', 'Unknown'),
                    "chunks": metadata.get('chunk_count', 0)
                }
        return list(documents.values())
    
    def clear_database(self):
        s = input('Are you sure you want to delete ALL documents? (y/n): ')

        if s.lower() == 'y':
            self.chroma_client.delete_collection(self.collection_name)

            db_path = self.persist_directory
            if os.path.exists(db_path):
                shutil.rmtree(db_path, ignore_errors=True)
                print(f"Folder deleted: {db_path}")

            os.makedirs(db_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=db_path, 
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                ))
            self.collection = self.get_or_create_collection_db()
            self.conversation_history = []

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:

        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start += chunk_size - overlap
        
        return chunks
    
    def _get_cahced_embedding(self, text: str):
        text_hash = hashlib.md5(text.encode())

        embedding = embedder.encode([text])[0]
        self.embeddings_cache[text_hash] = embedding

        return embedding
    
    def search(self, query: str, k: int = 3, filter_by_document: Optional[List] = None, filter_by_metadata: Optional[List] = None):
        # query_embedding = embedder.encode([query])[0]
        query_embedding = self._get_cahced_embedding(query)

        where_filter = None
        if filter_by_document:
            where_filter = {'document_id': filter_by_document}
        elif filter_by_metadata:
            where_filter = filter_by_metadata

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )

        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance_score': 1 - results['distances'][0][i]
            })

        return formatted_results
    
    def ask(self, command: str, k: int = 3, filter_by_document: Optional[str] = None) -> str:

        if self.collection.count() == 0:
            return {
                "answer": "No documents loaded. Please add documents first.",
                "sources": []
            }
        
        results = self.search(command, k=k, filter_by_document=filter_by_document)

        context_parts = []
        for i in results:
            doc_name = i['metadata'].get('document_name', 'Unknown')
            context_parts.append(f'[From: {doc_name}]\n{i['text']}')

        context = "\n\n---\n\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful assistant. Answer commands based on the provided context.
                If the answer is not in the context, say "I don't have that information in the document."

                Context:
                {context}
                """
            }
        ]
        for msg in self.conversation_history[-4:]:
            messages.append(msg)

        messages.append({"role": "user", "content": command})
        
        try:
            raw_response = llm_pipeline(messages)
            answer = raw_response[0]['generated_text']

            self.conversation_history.append({"role": "user", "content": command})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]
            
            return raw_response
        except:
            return f'Error generating response'
        
    def get_stats(self):
        return {
            'total_chunks': self.collection.count(),
            'collection_name': self.collection_name,
            'unique_documents': len(self.list_document())
        }
    
    def chat(self):
        print("\n" + "="*60)
        print("RAG Assistant Ready! Type 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                command = input("You: ").strip()

                if command.lower() == 'add':
                    url_file = input('Write the file url: ')

                    try:
                        self.add_documet(url_file)
                    except:
                        print(f'Failed tp file ({url_file})')

                if command.lower() == 'stats':
                    stats = self.get_stats()
                    print(f'Collection name: {stats['collection_name']}')
                    print(f'Number of chunks: {stats['total_chunks']}')
                    print(f'Unique documents: {stats['unique_documents']}')

                if command.lower() == 'list':
                    docs = self.list_document()
                    if docs:
                        for i, doc in enumerate(docs, 1):
                            print(f'{i}. {doc['name']}; {doc['chunks']}; {doc['added_date']}')
                            # print(f'{i}. {doc['name']}')
                            # print(f'   Number of chunks: {doc['chunks']}')
                            # print(f'   Date: {doc['added_date']}')
                    else:
                        print('No documents in database')

                if command.lower() in ['clear', 'clear database', 'database clear']:
                    self.clear_database()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if command.lower() == 'ask':               
                    question = input('Write your question: ').strip()

                    print("\nThinking...\n")
                    answer = self.ask(question)
                    print(f"Assistant: {answer[0]['generated_text']}\n")

                if not command:
                    print('Enter the command!')
                    continue
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():

    rag = ChromaRAG(persist_directory='./my_documents_db')
    
    if rag.collection.count() == 0:
        print('Database is empty')
    
    #     example_pdf = [
    #         'https://rus-center.lgaki.info/wp-content/uploads/2022/03/chehov_kryzhovnik.pdf',
    #         'https://old1.natlib.uz/Content/userfiles/upload/Дп%20стр/yubilyar/ru/130%20лет_Каштанка_Чехов%20Антон%20Павлович.pdf',
    #         'https://kurskmed.com/upload/departments/library/img/proekt-23/Chekhov.pdf'
    #     ]
        
    #     for url in example_pdf:
    #         try:
    #             rag.add_documet(url, 'url')
    #         except:
    #             print(f'Failed to add {url}')

    rag.chat()


if __name__ == "__main__":
    main()