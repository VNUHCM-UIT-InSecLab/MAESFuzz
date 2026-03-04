#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Helper for UniFuzz Fuzzing System
Tích hợp trực tiếp RAG vào fuzzing system thay vì sử dụng server
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add RAG directory to path
sys.path.append(str(Path(__file__).parent.parent / "RAG"))

try:
    # Add parent directory to path for RAG import
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from RAG.rag import LangChainRAG
    RAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RAG not available: {e}")
    RAG_AVAILABLE = False

logger = logging.getLogger(__name__)

class FuzzingRAGHelper:
    """Helper class để tích hợp RAG vào fuzzing system"""
    
    def __init__(self, use_gemini: bool = True):
        """
        Khởi tạo RAG helper
        Args:
            use_gemini: Sử dụng Google Gemini (cần GOOGLE_API_KEY)
        """
        self.rag_system = None
        self.use_gemini = use_gemini
        self.initialized = False
        
        if RAG_AVAILABLE:
            try:
                self.rag_system = LangChainRAG(use_gemini=use_gemini)
                # Load vector store và tạo QA chain
                self.rag_system._load_vector_store()
                if self.rag_system.vector_store is not None:
                    self.rag_system._create_qa_chain()
                self.initialized = True
                logger.info("RAG system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RAG system: {e}")
                self.rag_system = None
        else:
            logger.warning("RAG system not available")
    
    def get_vulnerability_insights(self, vulnerability_type: str, contract_name: str = None) -> Dict[str, Any]:
        """
        Lấy insights về vulnerability từ audit reports
        Args:
            vulnerability_type: Loại vulnerability (e.g., "integer overflow", "reentrancy")
            contract_name: Tên contract (optional)
        Returns:
            Dict chứa insights và recommendations
        """
        if not self.initialized or not self.rag_system:
            return {
                "status": "error",
                "message": "RAG system not available",
                "insights": [],
                "recommendations": []
            }
        
        try:
            # Tạo query để tìm kiếm thông tin về vulnerability
            query = f"vulnerability {vulnerability_type} smart contract security issue"
            if contract_name:
                query += f" {contract_name}"
            
            # Sử dụng RAG để tìm kiếm
            response = self.rag_system.qa_chain.invoke({"query": query})
            answer = response.get("result", "")
            
            # Trích xuất source documents
            source_docs = response.get("source_documents", [])
            
            insights = []
            recommendations = []
            
            # Phân tích source documents để trích xuất insights
            for doc in source_docs[:3]:  # Chỉ lấy top 3 documents
                content = doc.page_content
                source = doc.metadata.get('source', 'Unknown')
                
                # Tìm kiếm patterns trong content
                if vulnerability_type.lower() in content.lower():
                    insights.append({
                        "source": source,
                        "content": content[:500] + "..." if len(content) > 500 else content,
                        "relevance": "high"
                    })
                    
                    # Trích xuất recommendations từ content
                    if "recommend" in content.lower() or "fix" in content.lower() or "mitigate" in content.lower():
                        recommendations.append({
                            "source": source,
                            "recommendation": self._extract_recommendation(content)
                        })
            
            return {
                "status": "success",
                "vulnerability_type": vulnerability_type,
                "contract_name": contract_name,
                "answer": answer,
                "insights": insights,
                "recommendations": recommendations,
                "source_count": len(source_docs)
            }
            
        except Exception as e:
            logger.error(f"Error getting vulnerability insights: {e}")
            return {
                "status": "error",
                "message": str(e),
                "insights": [],
                "recommendations": []
            }
    
    def get_fuzzing_strategy(self, contract_name: str, functions: List[str], vulnerabilities: List[str]) -> Dict[str, Any]:
        """
        Lấy chiến lược fuzzing dựa trên contract và vulnerabilities
        Args:
            contract_name: Tên contract
            functions: Danh sách functions
            vulnerabilities: Danh sách vulnerabilities đã phát hiện
        Returns:
            Dict chứa chiến lược fuzzing
        """
        if not self.initialized or not self.rag_system:
            return {
                "status": "error",
                "message": "RAG system not available",
                "strategy": {}
            }
        
        try:
            # Tạo query cho fuzzing strategy
            query = f"fuzzing strategy smart contract {contract_name} functions {', '.join(functions)} vulnerabilities {', '.join(vulnerabilities)}"
            
            response = self.rag_system.qa_chain.invoke({"query": query})
            answer = response.get("result", "")
            
            # Tạo strategy recommendations
            strategy = {
                "contract_name": contract_name,
                "priority_functions": [],
                "test_patterns": [],
                "risk_assessment": {},
                "recommendations": []
            }
            
            # Phân tích functions để xác định priority
            for func in functions:
                if any(keyword in func.lower() for keyword in ['transfer', 'withdraw', 'deposit', 'mint', 'burn']):
                    strategy["priority_functions"].append({
                        "function": func,
                        "priority": "high",
                        "reason": "Financial operations"
                    })
                elif any(keyword in func.lower() for keyword in ['admin', 'owner', 'set', 'update']):
                    strategy["priority_functions"].append({
                        "function": func,
                        "priority": "medium",
                        "reason": "Administrative operations"
                    })
                else:
                    strategy["priority_functions"].append({
                        "function": func,
                        "priority": "low",
                        "reason": "General operations"
                    })
            
            # Phân tích vulnerabilities
            for vuln in vulnerabilities:
                if vuln.lower() in ['reentrancy', 'integer overflow', 'access control']:
                    strategy["risk_assessment"][vuln] = "high"
                elif vuln.lower() in ['gas optimization', 'unchecked return']:
                    strategy["risk_assessment"][vuln] = "medium"
                else:
                    strategy["risk_assessment"][vuln] = "low"
            
            # Tạo recommendations
            if strategy["risk_assessment"].get("high"):
                strategy["recommendations"].append("Focus on high-risk vulnerabilities with extensive testing")
            
            if len(strategy["priority_functions"]) > 0:
                strategy["recommendations"].append("Prioritize testing of financial and administrative functions")
            
            strategy["recommendations"].append("Use boundary value testing for integer operations")
            strategy["recommendations"].append("Test with malicious external contracts for reentrancy")
            
            return {
                "status": "success",
                "strategy": strategy,
                "ai_insights": answer
            }
            
        except Exception as e:
            logger.error(f"Error getting fuzzing strategy: {e}")
            return {
                "status": "error",
                "message": str(e),
                "strategy": {}
            }
    
    def get_test_case_suggestions(self, function_name: str, function_signature: str, vulnerability_context: str = None) -> List[Dict[str, Any]]:
        """
        Lấy suggestions cho test cases dựa trên function và vulnerability context
        Args:
            function_name: Tên function
            function_signature: Signature của function
            vulnerability_context: Context về vulnerability (optional)
        Returns:
            List các test case suggestions
        """
        if not self.initialized or not self.rag_system:
            return []
        
        try:
            query = f"test cases smart contract function {function_name} {function_signature}"
            if vulnerability_context:
                query += f" vulnerability {vulnerability_context}"
            
            response = self.rag_system.qa_chain.invoke({"query": query})
            answer = response.get("result", "")
            
            # Tạo test case suggestions dựa trên function type
            suggestions = []
            
            # Boundary value testing
            if "uint" in function_signature.lower():
                suggestions.extend([
                    {"type": "boundary", "value": 0, "description": "Zero value test"},
                    {"type": "boundary", "value": 1, "description": "Minimal positive value"},
                    {"type": "boundary", "value": 2**256 - 1, "description": "Max uint256 value"},
                    {"type": "boundary", "value": 2**128, "description": "Large value test"}
                ])
            
            # Address testing
            if "address" in function_signature.lower():
                suggestions.extend([
                    {"type": "address", "value": "0x0000000000000000000000000000000000000000", "description": "Zero address test"},
                    {"type": "address", "value": "0x1111111111111111111111111111111111111111", "description": "Non-zero address test"}
                ])
            
            # Reentrancy testing
            if vulnerability_context and "reentrancy" in vulnerability_context.lower():
                suggestions.append({
                    "type": "reentrancy",
                    "value": "recursive_call",
                    "description": "Test recursive external call"
                })
            
            return suggestions[:10]  # Trả về tối đa 10 suggestions
            
        except Exception as e:
            logger.error(f"Error getting test case suggestions: {e}")
            return []
    
    def _extract_recommendation(self, content: str) -> str:
        """Trích xuất recommendation từ content"""
        lines = content.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'fix', 'mitigate']):
                return line.strip()
        return content[:100] + "..." if len(content) > 100 else content
    
    def is_available(self) -> bool:
        """Kiểm tra xem RAG system có available không"""
        return self.initialized and self.rag_system is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê về RAG system"""
        if not self.initialized or not self.rag_system:
            return {"status": "not_available"}
        
        try:
            return self.rag_system.get_stats()
        except Exception as e:
            return {"status": "error", "message": str(e)}


def create_fuzzing_rag_helper(use_gemini: bool = True) -> FuzzingRAGHelper:
    """
    Factory function để tạo FuzzingRAGHelper
    Args:
        use_gemini: Sử dụng Google Gemini
    Returns:
        FuzzingRAGHelper instance
    """
    return FuzzingRAGHelper(use_gemini=use_gemini)


# Test function
if __name__ == "__main__":
    # Test RAG helper
    rag_helper = create_fuzzing_rag_helper()
    
    if rag_helper.is_available():
        print("✅ RAG Helper initialized successfully")
        
        # Test vulnerability insights
        insights = rag_helper.get_vulnerability_insights("integer overflow", "TestContract")
        print(f"Vulnerability insights: {insights['status']}")
        
        # Test fuzzing strategy
        strategy = rag_helper.get_fuzzing_strategy(
            "TestContract", 
            ["transfer", "withdraw", "deposit"], 
            ["integer overflow", "reentrancy"]
        )
        print(f"Fuzzing strategy: {strategy['status']}")
        
        # Test test case suggestions
        suggestions = rag_helper.get_test_case_suggestions(
            "transfer", 
            "function transfer(address to, uint256 amount)", 
            "reentrancy"
        )
        print(f"Test case suggestions: {len(suggestions)}")
        
    else:
        print("❌ RAG Helper not available")
