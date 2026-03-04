#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hệ thống RAG thông minh cho UniFuzz
Sử dụng LangChain với FAISS persistent storage
Dựa trên: https://python.plainenglish.io/chatting-with-pdfs-building-a-simple-rag-system-6d3617ae77d3
"""

import os
import json
import logging
import sys
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from pathlib import Path
import hashlib
from dataclasses import dataclass

# LangChain imports
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# OCR và xử lý hình ảnh
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io

# PDF processing
import pdfplumber
import PyPDF2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class AuditFinding:
    """Cấu trúc dữ liệu cho audit finding"""
    title: str
    severity: str
    description: str
    impact: str
    recommendation: str
    code_snippet: str
    source_file: str
    page_number: int

class LangChainRAG:
    """Hệ thống RAG sử dụng LangChain với FAISS persistent storage"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_db_path: str = "./vector_db",
                 use_gemini: bool = True):
        """
        Khởi tạo LangChain RAG
        Args:
            embedding_model: Model embedding từ HuggingFace
            vector_db_path: Đường dẫn lưu FAISS vector database
            use_gemini: Sử dụng Google Gemini LLM (cần GOOGLE_API_KEY)
        """
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(exist_ok=True)
        
        # FAISS index path
        self.faiss_index_path = self.vector_db_path / "faiss_index"
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Sử dụng CPU để tránh lỗi CUDA
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Vector store
        self.vector_store = None
        
        # RAG QA chain
        self.qa_chain = None
        
        # OCR configuration
        self.ocr_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}/-+= '
        
        # Audit reports directory
        self.audit_reports_dir = Path("./RAG/data/audit_reports")
        
        # Gemini option
        self.use_gemini = use_gemini
        
        logger.info("LangChain RAG initialized successfully")

    def ingest(self, max_files: int = 10, specific_file: Optional[str] = None):
        """
        Ingest audit reports vào FAISS vector database
        Args:
            max_files: Số lượng files tối đa để process
            specific_file: File cụ thể để process (optional)
        """
        if not self.audit_reports_dir.exists():
            logger.warning(f"Audit reports directory not found: {self.audit_reports_dir}")
            return
        
        # Load existing vector store nếu có
        self._load_vector_store()
        
        # Process files
        if specific_file:
            pdf_files = [Path(specific_file)]
            if not pdf_files[0].exists():
                logger.error(f"File not found: {specific_file}")
                return
        else:
            pdf_files = list(self.audit_reports_dir.glob("*.pdf"))[:max_files]
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        # Process từng file
        all_documents = []
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing: {pdf_file.name}")
                documents = self._process_pdf_file(pdf_file)
                all_documents.extend(documents)
                logger.info(f"Extracted {len(documents)} chunks from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
        
        if not all_documents:
            logger.warning("No documents extracted from PDFs")
            return
        
        # Tạo hoặc cập nhật vector store
        if self.vector_store is None:
            logger.info("Creating new FAISS vector store...")
            self.vector_store = FAISS.from_documents(all_documents, self.embeddings)
        else:
            logger.info("Updating existing FAISS vector store...")
            self.vector_store.add_documents(all_documents)
        
        # Lưu vector store
        self._save_vector_store()
        
        # Tạo RAG QA chain
        self._create_qa_chain()
        
        logger.info(f"Ingest completed: {len(all_documents)} documents processed")

    def ask(self, query: str, k: int = 5):
        """
        Ask question về audit findings
        Args:
            query: Câu hỏi
            k: Số lượng documents để retrieve
        """
        # Try to load vector store first
        self._load_vector_store()
        
        if self.qa_chain is None:
            print("❌ No vector database available. Please run 'ingest' first.")
            return
        
        print(f"🔍 Question: {query}")
        print("=" * 60)
        
        try:
            # Sử dụng RAG QA chain
            response = self.qa_chain.invoke({"query": query})
            answer = response.get("result", "No answer found")
            
            # Xác định LLM đang sử dụng
            llm_type = "Gemini" if self.use_gemini and os.getenv("GOOGLE_API_KEY") else "Fallback"
            
            print(f"🤖 AI Answer ({llm_type}):")
            print("=" * 60)
            print(answer)
            print()
            
            # Hiển thị source documents
            if "source_documents" in response:
                print(f"📚 Retrieved Documents ({len(response['source_documents'])}):")
                print("=" * 60)
                for i, doc in enumerate(response["source_documents"][:k], 1):
                    source_file = doc.metadata.get('source', 'Unknown')
                    if 'RAG/data/audit_reports/' in source_file:
                        filename = source_file.split('/')[-1]
                    else:
                        filename = source_file
                    
                    print(f"{i}. 📄 {filename}")
                    print(f"   📖 Page: {doc.metadata.get('page', 'N/A')}")
                    print(f"   📝 Content Preview:")
                    
                    # Hiển thị content tốt hơn
                    content = doc.page_content.strip()
                    if len(content) > 300:
                        content = content[:300] + "..."
                    
                    # Format content với indentation
                    lines = content.split('\n')
                    for line in lines[:5]:  # Chỉ hiển thị 5 dòng đầu
                        if line.strip():
                            print(f"      {line.strip()}")
                    
                    print()
            
        except Exception as e:
            logger.error(f"Error during QA: {e}")
            print(f"❌ Error: {e}")
            print("💡 Tip: Make sure you have set GOOGLE_API_KEY environment variable")

    def _process_pdf_file(self, pdf_path: Path) -> List[Document]:
        """Process một PDF file và trả về documents"""
        documents = []
        
        try:
            # Thử load với PyPDFLoader trước
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                
                # Nếu có ít text, thử OCR
                if len(docs) == 0 or sum(len(doc.page_content.strip()) for doc in docs) < 500:
                    docs = self._process_pdf_with_ocr(pdf_path)
                
            except Exception:
                # Fallback to OCR processing
                docs = self._process_pdf_with_ocr(pdf_path)
            
            # Split documents thành chunks
            for doc in docs:
                chunks = self.text_splitter.split_documents([doc])
                documents.extend(chunks)
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
        
        return documents

    def _process_pdf_with_ocr(self, pdf_path: Path) -> List[Document]:
        """Process PDF với OCR"""
        documents = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(min(len(doc), 20)):  # Giới hạn 20 trang
                page = doc.load_page(page_num)
                
                # Lấy text thông thường
                text = page.get_text()
                
                # Nếu text ít, dùng OCR
                if len(text.strip()) < 100:
                    # Convert page thành image
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # OCR
                    ocr_text = pytesseract.image_to_string(
                        Image.open(io.BytesIO(img_data)),
                        config=self.ocr_config
                    )
                    
                    if len(ocr_text) > len(text):
                        text = ocr_text
                
                # Tạo document nếu có nội dung
                if len(text.strip()) > 50:
                    doc_obj = Document(
                        page_content=text,
                        metadata={
                            "source": str(pdf_path),
                            "page": page_num + 1,
                            "file_name": pdf_path.name
                        }
                    )
                    documents.append(doc_obj)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"OCR processing error for {pdf_path}: {e}")
        
        return documents

    def _load_vector_store(self):
        """Load FAISS vector store từ disk"""
        faiss_files = list(self.faiss_index_path.glob("*.faiss"))
        if faiss_files:
            try:
                logger.info("Loading existing FAISS vector store...")
                self.vector_store = FAISS.load_local(
                    str(self.faiss_index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self._create_qa_chain()
                logger.info("Vector store loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load vector store: {e}")
                self.vector_store = None

    def _save_vector_store(self):
        """Save FAISS vector store ra disk"""
        if self.vector_store is not None:
            try:
                self.vector_store.save_local(str(self.faiss_index_path))
                logger.info(f"Vector store saved to {self.faiss_index_path}")
            except Exception as e:
                logger.error(f"Could not save vector store: {e}")

    def _create_qa_chain(self):
        """Tạo RAG QA chain với Google Gemini hoặc fallback"""
        if self.vector_store is not None:
            if self.use_gemini and os.getenv("GOOGLE_API_KEY"):
                try:
                    # Sử dụng Google Gemini LLM
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    
                    # Khởi tạo Gemini LLM
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        temperature=0.7,
                        max_tokens=1024,
                        google_api_key=os.getenv("GOOGLE_API_KEY")
                    )
                    
                    self.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=self.vector_store.as_retriever(
                            search_kwargs={"k": 5}
                        ),
                        return_source_documents=True
                    )
                    
                    logger.info("QA chain created successfully with Google Gemini")
                    return
                except Exception as e:
                    logger.warning(f"Could not create QA chain with Gemini: {e}")
                    logger.info("Falling back to simple retrieval...")
            
            # Fallback to simple retrieval without LLM
            self._create_simple_retrieval()

    def _create_simple_retrieval(self):
        """Fallback method khi không có LLM"""
        try:
            from langchain_community.llms import FakeListLLM
            
            fake_llm = FakeListLLM(
                responses=["Based on the retrieved documents, I found relevant information but cannot provide a detailed analysis without a proper LLM."]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=fake_llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 5}
                ),
                return_source_documents=True
            )
            
            logger.info("QA chain created with fallback LLM")
        except Exception as e:
            logger.error(f"Could not create fallback QA chain: {e}")

    def get_stats(self):
        """Lấy thống kê về vector database"""
        # Try to load vector store first
        self._load_vector_store()
        
        if self.vector_store is None:
            print("❌ No vector database available")
            return
        
        try:
            # Lấy thông tin cơ bản
            print("📊 Vector Database Statistics")
            print("=" * 40)
            print(f"Vector DB Path: {self.vector_db_path}")
            print(f"FAISS Index Path: {self.faiss_index_path}")
            print(f"Index exists: {self.faiss_index_path.exists()}")
            
            # Thống kê files
            if self.audit_reports_dir.exists():
                pdf_files = list(self.audit_reports_dir.glob("*.pdf"))
                print(f"Available PDF files: {len(pdf_files)}")
                
                if pdf_files:
                    print("Recent files:")
                    for pdf_file in sorted(pdf_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                        print(f"  - {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")


def main():
    """Main function với argparse command line interface"""
    parser = argparse.ArgumentParser(description="LangChain RAG System for Smart Contract Audit Reports with Google Gemini")
    parser.add_argument("--no-gemini", action="store_true", help="Disable Google Gemini (use fallback mode)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest audit reports into vector database")
    ingest_parser.add_argument("--max-files", type=int, default=10, help="Maximum number of files to process")
    ingest_parser.add_argument("--file", type=str, help="Specific file to process")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask questions about audit findings")
    ask_parser.add_argument("query", nargs="+", help="Question to ask")
    ask_parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show vector database statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize RAG
    use_gemini = not args.no_gemini
    rag = LangChainRAG(use_gemini=use_gemini)
    
    if args.command == "ingest":
        rag.ingest(max_files=args.max_files, specific_file=args.file)
    
    elif args.command == "ask":
        query = " ".join(args.query)
        rag.ask(query, k=args.k)
    
    elif args.command == "stats":
        rag.get_stats()


if __name__ == "__main__":
    main()