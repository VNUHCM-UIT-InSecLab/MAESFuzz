#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Cài đặt danh sách các components có sẵn
from .individual import Individual
from .population import Population
from .generator import Generator

# Import RAG enhanced components 
from .rag_enhanced_generator import RAGEnhancedGenerator, create_rag_enhanced_generator
from .rag_enhanced_population import RAGEnhancedPopulation

# Import LLM related components
try:
    from .llm_agent import LLMAgent
    from .llm_enhanced_generator import LLMEnhancedGenerator, create_llm_enhanced_generator
    has_llm_support = True
except ImportError:
    # Fallback nếu không có LLM support
    has_llm_support = False
    
# Xuất các components được sử dụng trong project
__all__ = [
    'Individual', 
    'Population', 
    'Generator',
    'RAGEnhancedGenerator', 
    'create_rag_enhanced_generator',
    'RAGEnhancedPopulation'
]

# Thêm LLM components vào exports nếu có
if has_llm_support:
    __all__.extend(['LLMAgent', 'LLMEnhancedGenerator', 'create_llm_enhanced_generator'])