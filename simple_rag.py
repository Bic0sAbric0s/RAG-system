import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
import requests
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()


model_name = "Qwen/Qwen2.5-1.5B-Instruct"

embedder = SentenceTransformer('intfloat/multilingual-e5-large')
print("Model loaded!")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
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
    """Minimal RAG implementation - in-memory only (no database)"""
    
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.conversation_history = []
    
    def load_pdf_from_url(self, url: str):
        """Load and process PDF"""
        print(f"Downloading PDF from {url}...")
        response = requests.get(url)
        pdf_file = BytesIO(response.content)
        
        # Extract text
        print("Extracting text...")
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Chunk the text
        print("Chunking text...")
        self.chunks = self._chunk_text(text, chunk_size=800, overlap=100)
        print(f"Created {len(self.chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings = embedder.encode(self.chunks, show_progress_bar=True)
        print("Ready to answer questions!")
    
    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
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
        """Find most relevant chunks for the question"""
        # Generate embedding for question
        question_embedding = embedder.encode([question])[0]
        
        # Calculate cosine similarity
        similarities = []
        for chunk_embedding in self.embeddings:
            similarity = np.dot(question_embedding, chunk_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append(similarity)
        
        # Get top k chunks
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.chunks[i] for i in top_indices]
    
    def ask(self, question: str) -> str:
        """Ask a question and get an answer"""
        # Find relevant context
        relevant_chunks = self._find_relevant_chunks(question)
        context = "\n\n---\n\n".join(relevant_chunks)
        
        # Build messages
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
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        try:
            raw_response = llm_pipeline(messages)
            answer = raw_response[0]['generated_text']
            # print(f"Чистый ответ: {raw_response[0]['generated_text']}")
            

            # Save to history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            # Keep only last 6 messages (3 turns)
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]
            
            return raw_response
        except:
            return f'Error generating response'
    
    def chat(self):
        """Interactive chat loop"""
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
    """Run the RAG assistant"""
    
    rag = SimpleRAG()
    
    pdf_url = "https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
    rag.load_pdf_from_url(pdf_url)
    
    rag.chat()


if __name__ == "__main__":
    main()