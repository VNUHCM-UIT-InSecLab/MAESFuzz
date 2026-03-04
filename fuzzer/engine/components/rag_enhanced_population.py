#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import json
import logging
from typing import Dict, List, Any, Optional

from fuzzer.utils.utils import initialize_logger
from .population import Population

class RAGEnhancedPopulation(Population):
    """
      size = max(2 * số_hàm_trong_interface, 10)
    """

    def __init__(self, indv_template, indv_generator, size=None, other_generators=None):
        # Tự động tính size nếu không được truyền từ engine
        if size is None:
            interface = getattr(indv_generator, "interface", {}) or {}
            func_count = len(interface)
            size = max(2 * func_count, 10)

        super().__init__(indv_template, indv_generator, size, other_generators)
        self.logger = initialize_logger("RAGEnhancedPopulation")
        self.logger.info("Initialized RAGEnhancedPopulation with size=%d", self.size)
    
    def init(self, init_seed=True, **kwargs):
        """
        Initialize population with individuals.
        """
        self.logger.info(f"Initializing population with {self.size} individuals")

        # Allow extra kwargs (e.g., no_cross) to be forwarded to base init
        result = super().init(init_seed=init_seed, **kwargs)

        self.logger.info(f"RAGEnhancedPopulation initialized with {len(self.individuals)} individuals")
        return result 