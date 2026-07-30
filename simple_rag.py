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
from dotenv import load_dotenv

load_dotenv()


client = chromadb.Client()
collection = client.get_or_create_collection(
    name='',
    metadata={"hnsw:space": "cosine"}
)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

embedder = SentenceTransformer('intfloat/multilingual-e5-large')
print("Model loaded!")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

llm_pipeline = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
    repetition_penalty=1.1,
    return_full_text=False
)

class SimpleRAG:    
    def __init__(self, persist_directory='./chroma_db'):
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

        self.chunks = []
        self.embeddings = None
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
        for page in reader.pages:
            text += page.extract_text() + '\n'

        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")

        print("Chunking text...")
        self.chunks = self._chunk_text(text, chunk_size=800, overlap=100)
        print(f"Created {len(self.chunks)} chunks")
        

        print("Generating embeddings...")
        self.embeddings = embedder.encode(self.chunks, show_progress_bar=True)
        print("Ready to answer questions!")

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
            
        self.collection.add(
            embeddings=self.embeddings.tolist(),
            documents=self.chunks,
        )

        self.document_registry[doc_id] = {
            "name": doc_name,
            "chunks": len(self.chunks),
            "source": source,
            "added_date": base_metadata["added_date"]
        }
        print(f'Document {doc_name} added')

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:

        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start += chunk_size - overlap
        
        return chunks
    
    def _find_relevant_chunks(self, question: str, k: int = 3) -> List[str]:

        question_embedding = embedder.encode([question])[0]
        
        similarities = []
        for chunk_embedding in self.embeddings:
            similarity = np.dot(question_embedding, chunk_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append(similarity)

        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.chunks[i] for i in top_indices]
    
    def ask(self, question: str) -> str:

        relevant_chunks = self._find_relevant_chunks(question)
        context = "\n\n---\n\n".join(relevant_chunks)

        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful assistant. Answer questions based on the provided context.
                If the answer is not in the context, say "I don't have that information in the document."

                Context:
                {context}
                """
            }
        ]
        
        messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": question})
        
        try:
            raw_response = llm_pipeline(messages)
            answer = raw_response[0]['generated_text']

            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]
            
            return raw_response
        except:
            return f'Error generating response'
    
    def chat(self):
        print("\n" + "="*60)
        print("RAG Assistant Ready! Type 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\nThinking...\n")
                answer = self.ask(question)
                print(f"Assistant: {answer[0]['generated_text']}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():

    rag = SimpleRAG()
    
    pdf_url = "https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
    rag.load_pdf_from_url(pdf_url)
    
    rag.chat()


if __name__ == "__main__":
    main()