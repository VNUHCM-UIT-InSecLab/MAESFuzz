import os
import argparse
import logging
import time
import random
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile
from func_timeout import func_timeout, FunctionTimedOut

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RAG:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=self.api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=256
        )
        self.vector_db_path = "./RAG/data/vector_db"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 4  # 4 seconds between requests (15 RPM)
        self.request_count = 0
        self.max_requests_per_minute = 10  # Conservative limit

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed + random.uniform(0.5, 1.5)
            logging.info(f"Rate limiting: waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1

    def extract_text_from_image(self, image):
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            logging.error(f"OCR Error: {e}")
            return ""

    def process_pdf_page(self, page, pdf_path, page_num):
        text = page.extract_text() or ""
        if len(text.strip()) < 50:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    images = convert_from_path(
                        pdf_path,
                        first_page=page_num,
                        last_page=page_num,
                        output_folder=temp_dir,
                        fmt="jpeg"
                    )
                    for img in images:
                        text += "\n" + self.extract_text_from_image(img)
            except Exception as e:
                logging.error(f"PDF to image conversion error: {e}")
        return text   

    def ingest(self, pdf_dir="./RAG/data/audit_reports"):
        all_text = ""
        
        if not os.path.exists(pdf_dir):
            logging.warning(f"Directory {pdf_dir} does not exist")
            return
            
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            logging.warning(f"No PDF files found in {pdf_dir}")
            return

        logging.info(f"Processing {len(pdf_files)} PDF files...")
        
        for i, filename in enumerate(pdf_files[:10]):  # Limit to first 10 files to avoid timeout
            file_path = os.path.join(pdf_dir, filename)
            logging.info(f"Processing {i+1}/{min(10, len(pdf_files))}: {filename}")
            
            try:
                with open(file_path, "rb") as f:
                    pdf_reader = PdfReader(f)
                    for page_num, page in enumerate(pdf_reader.pages[:5]):  # Limit pages per PDF
                        try:
                            processed_text = self.process_pdf_page(
                                page, 
                                file_path, 
                                page_num + 1
                            )
                            all_text += processed_text + "\n"
                        except Exception as e:
                            logging.error(f"Error processing page {page_num+1} in {filename}: {e}")
            except Exception as e:
                logging.error(f"Error opening {filename}: {e}")

        if not all_text.strip():
            logging.error("No text could be extracted from PDFs")
            return

        try:
            chunks = self.text_splitter.split_text(all_text)
            logging.info(f"Creating vector store with {len(chunks)} chunks...")
            vector_store = FAISS.from_texts(chunks, self.embeddings)
            vector_store.save_local(self.vector_db_path)
            logging.info(f"Vector store updated with {len(chunks)} chunks")
        except Exception as e:
            logging.error(f"Vector store error: {e}")

    def ask_simple(self, question):
        """Simple ask without LLM processing to avoid rate limits"""
        try:
            vector_store = FAISS.load_local(
                self.vector_db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            logging.info(f"Vector store loaded with {vector_store.index.ntotal} vectors")
        except Exception as e:
            logging.error(f"Error loading vector database: {e}")
            return "No relevant information available"

        try:
            # Simple similarity search - NO LLM calls
            docs = vector_store.similarity_search(question, k=3)
            if not docs:
                return "No relevant information available"
            
            # Return concatenated context
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                 for i, doc in enumerate(docs)])
            
            return f"Based on audit reports:\n\n{context}"
            
        except Exception as e:
            logging.error(f"Error in similarity search: {e}")
            return "No relevant information available"

    def ask(self, question):
        """Full RAG with rate limiting"""
        # Check if we should use simple mode to avoid rate limits
        if self.request_count > self.max_requests_per_minute:
            logging.warning("Switching to simple mode due to rate limits")
            return self.ask_simple(question)
            
        try:
            vector_store = FAISS.load_local(
                self.vector_db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logging.error(f"Error loading database: {e}")
            return "No relevant information available"

        # SIMPLIFIED APPROACH - NO QUESTION DECOMPOSITION
        # NO COMPRESSION RETRIEVER to reduce API calls
        
        try:
            # Simple retrieval - only 1 potential API call
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(question)
            
            if not docs:
                return "No relevant information available"
                
        except Exception as e:
            logging.error(f"Error in retrieval: {e}")
            return "No relevant information available"

        # Simple prompt without complex processing
        prompt_template = (
            "Answer based on the audit report context below. "
            "If no relevant information, say 'No relevant information available'.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer briefly and focus on key vulnerabilities:"
        )

        context = "\n".join([doc.page_content for doc in docs])
        
        try:
            # Rate limit before API call
            self._wait_for_rate_limit()
            
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                google_api_key=self.api_key
            )

            response = func_timeout(
                30,
                model.invoke,
                args=(prompt_template.format(context=context, question=question),)
            )
            
            print("\nResponse from AI:")
            print(response.content)
            return response.content
            
        except FunctionTimedOut:
            logging.error("Request timed out")
            return "Request timed out"
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                logging.warning("Rate limit hit, falling back to simple mode")
                return self.ask_simple(question)
            else:
                logging.error(f"Error in query: {e}")
                return "No relevant information available"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG with rate limiting")
    parser.add_argument("command", choices=["ingest", "ask", "ask-simple"], help="Command")
    parser.add_argument("--api-key", help="Google API Key")
    parser.add_argument("--question", help="Question for ask command")
    
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logging.error("API Key not found")
        exit(1)

    rag = RAG(api_key)
    
    if args.command == "ingest":
        rag.ingest()
    elif args.command == "ask":
        if not args.question:
            logging.error("Ask command requires --question parameter.")
            exit(1)
        rag.ask(args.question)
    elif args.command == "ask-simple":
        if not args.question:
            logging.error("Ask command requires --question parameter.")
            exit(1)
        result = rag.ask_simple(args.question)
        print("\nResponse:")
        print(result)
