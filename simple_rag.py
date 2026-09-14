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
import docx
import requests
from io import BytesIO
import shutil
import hashlib
from pathlib import Path
import json
import time


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=True
)

# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

embedder = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
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
    temperature=0.3,
    do_sample=True,       # попробовать изменить
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
        self.cache_dir = Path('./cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'responses.json'  
        
        if not self.cache_file.exists():
            self.cache_file.write_text('{}', encoding='utf-8')
        self._load_cache()
        

        self.chunks = []
        self.embeddings = None
        self.embeddings_cache = {}
        self.conversation_history = []

        print(f'Documents in DB {self.collection.count()}')

    def _load_cache(self):
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.response_cache = json.load(f)
        except FileNotFoundError:
            self.response_cache = {}
            self._save_cache()

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.response_cache, f)

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
        
    def get_last_document_id(self) -> Optional[str]:

        if not self.collection:
            return None
        
        results = self.collection.get(include=["metadatas"])
        
        if not results['metadatas']:
            print("📭 База данных пуста")
            return None
        
        documents = {}
        for metadata in results['metadatas']:
            doc_id = metadata.get('document_id')
            doc_name = metadata.get('document_name', 'Unknown')
            added_date = metadata.get('added_date', '')
            
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    'id': doc_id,
                    'name': doc_name,
                    'date': added_date,
                    'chunk_count': 0
                }
            
            if doc_id in documents:
                documents[doc_id]['chunk_count'] += 1
        
        # Находим последний документ по дате
        last_doc = None
        last_date = ''
        
        for doc_id, doc_info in documents.items():
            if doc_info['date'] > last_date:
                last_date = doc_info['date']
                last_doc = doc_info
        
        return last_doc

        
    def delete_last_document(self):

        # if not self.collection:
        #     return False, "База данных не подключена"

        last_doc = self.get_last_document_id()
        
        if not last_doc:
            return False, "Документы не найдены в базе"
        
        doc_id = last_doc['id']
        doc_name = last_doc['name']

        results = self.collection.get(
            where={"document_id": doc_id},
            include=["metadatas"]
        )
        
        if not results['ids']:
            return False, f"Чанки документа '{doc_name}' не найдены"

        self.collection.delete(ids=results['ids'])
        
        return True, f"Документ '{doc_name}' удален ({len(results['ids'])} чанков)"
            
    
    def add_documet(self, source: str, source_type: str = 'url', file_bytes = None, metadata: Optional[Dict] = None) -> str:
        doc_id = str(uuid.uuid4())
        
        if source_type == 'bytes' and file_bytes is not None:
            doc_name = source or "document"
            text = self.filtering_files_by_extensions(file_bytes, doc_name)

        elif source_type == 'url':
            response = requests.get(source)
            doc_name = source.split('/')[-1]
            text = self.filtering_files_by_extensions(BytesIO(response.content), doc_name)
        
        elif source_type == 'file':
                doc_name = os.path.basename(source)
                with open(source, 'rb') as f:
                    file_bytes = f.read()
                text = self.filtering_files_by_extensions(file_bytes, doc_name)
        

        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")

        print("Chunking text...")
        self.chunks = self._chunk_text(text, chunk_size=800, overlap=100)
        print(f"Created {len(self.chunks)} chunks")
        

        print("Generating embeddings...")
        self.embeddings = embedder.encode(self.chunks, batch_size=128, show_progress_bar=True, device=device)
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

        self.documet_registry[doc_id] = {
            "name": doc_name,
            "chunks": len(self.chunks),
            "source": source,
            "added_date": base_metadata["added_date"]
        }
        print(f'Document {doc_name} added')
        return doc_id
    
    def filtering_files_by_extensions(self, file_bytes, doc_name):
        extension = os.path.splitext(doc_name)[1].lower()

        if extension == '.pdf':
            return self.pdf_extension(file_bytes)
        elif extension == '.docx':
            return self.docx_extension(file_bytes)
        elif extension == '.txt':
            return self.txt_extension(file_bytes)

    def pdf_extension(self, file_bytes):
        pdf_file = BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text
    
    def docx_extension(self, file_bytes):
        docx_file = BytesIO(file_bytes)
        doc = docx.Document(docx_file)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text
    
    def txt_extension(self, file_bytes):
        return file_bytes.decode('utf-8', errors='ignore')
    
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
            if end < len(text):
                search_start = start + int(chunk_size * 0.5)
                best_split = -1
                
                for sep in ['. ', '! ', '? ', '.\n', '\n\n', '\n']:
                    pos = text.rfind(sep, search_start, end)
                    if pos > best_split:
                        best_split = pos + len(sep)
                
                if best_split > 0:
                    end = best_split
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = max(end - overlap, start + 1)
        
        return chunks
    
    def _get_cahced_embedding(self, text: str):
        text_hash = hashlib.md5(text.encode())

        embedding = embedder.encode([text])[0]
        self.embeddings_cache[text_hash] = embedding

        return embedding
    
    def search(self, query: str, k: int = 3, filter_by_document: Optional[List] = None, filter_by_metadata: Optional[List] = None):
        min_relevance = 0.5
        query_embedding = self._get_cahced_embedding(query)

        where_filter = None
        if filter_by_document:
            where_filter = {'document_id': filter_by_document}
        elif filter_by_metadata:
            where_filter = filter_by_metadata

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k*3, 15),
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )

        candidates = []
        formatted_results = []
        query_words = set(query.lower().split())

        # for i in range(len(results['documents'][0])):
        #     formatted_results.append({
        #         'text': results['documents'][0][i],
        #         'metadata': results['metadatas'][0][i],
        #         'distance': results['distances'][0][i],
        #         'relevance_score': 1 - results['distances'][0][i]
        #     })

        #     # if int(formatted_results['relevance_score']) < min_relevance:
        #     #     continue

        #     text_words = set(formatted_results['text'].lower().split())
        #     overlap = len(query_words & text_words)
        #     length_penalty = 1.0 / (1 + len(formatted_results['text']) / 1000)

        #     final_score = int(formatted_results['relevance_score']) + 0.1 * overlap * length_penalty
            
        #     candidates.append({
        #         'text': formatted_results['text'],
        #         'metadata': results['metadatas'][0][i],
        #         'distance': formatted_results['distance'],
        #         'relevance_score': formatted_results['relevance_score'],
        #         'final_score': final_score
        #     })
        
        # candidates.sort(key=lambda x: x['final_score'], reverse=True)

        # formatted_results = candidates[:k]
        
        # if not formatted_results and candidates:
        #     formatted_results = candidates[:1]

        return formatted_results
    
    def ask(self, command: str, k: int = 3, filter_by_document: Optional[str] = None) -> str:

        if self.collection.count() == 0:
            return "Добавьте первый документ. База данных пуста."
                
        
        results = self.search(command, k=k, filter_by_document=filter_by_document)

        cache_key = hashlib.md5(command.encode()).hexdigest()
        if cache_key in self.response_cache:
            print(self.response_cache[cache_key])
            return str(self.response_cache[cache_key])

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
        
        # try:
        raw_response = llm_pipeline(messages)
        answer = raw_response[0]['generated_text']

        self.conversation_history.append({"role": "user", "content": command})
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]

        self.response_cache[cache_key] = answer
        self._save_cache()
        
        return answer
        # except:
        #     return f'Error generating response'
        
    def get_stats(self):
        return {
            'total_chunks': self.collection.count(),
            'collection_name': self.collection_name,
            'unique_documents': len(self.list_document())
        }
    
    def clear_cache(self):
        self.response_cache = {}
        self._save_cache()
        return '✅ Готово!'
    
    def chat(self):
        print("\n" + "="*60)
        print("RAG Assistant Ready! Type 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                command = input("You: ").strip()

                if command.lower() == 'response cache':
                    print(self.response_cache)

                if command.lower() == 'cache':
                    print({
                        'size': len(self.response_cache),
                        'file_size': self.cache_file.stat().st_size                
                        })

                if command.lower() in ['clear cache', 'cache clear', 'cache cl', 'cache c']:
                    self.clear_cache()

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
                    print(f"Assistant: {answer}\n")

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