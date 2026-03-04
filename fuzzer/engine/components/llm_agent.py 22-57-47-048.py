#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import time
import re
import random
import requests
from typing import Dict, List, Any, Optional, Union, Tuple

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LLMAgent")

class LLMAgent:
    """
    Agent sử dụng RAG để sinh và tối ưu các giá trị cho fuzzing
    """
    
    def __init__(self, api_key: str, api_endpoint: str = "http://localhost:5000/request"):
        """
        Khởi tạo LLMAgent sử dụng RAG qua API Flask
        
        :param api_key: Google API Key (vẫn giữ để tương thích với RAG)
        :param api_endpoint: Địa chỉ API của Flask server
        """
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.cache = {}  # Cache kết quả để tránh gọi API nhiều lần
        
        # Khởi tạo type mapping
        self.type_mapping = {
            "bool": bool,
            "address": str,
            "string": str,
            "bytes": bytes
        }
        
        # Khởi tạo các pattern regex
        self.uint_pattern = re.compile(r"uint(\d+)")
        self.int_pattern = re.compile(r"int(\d+)")
        self.bytes_pattern = re.compile(r"bytes(\d+)")
        self.array_pattern = re.compile(r"(.*)\[\]")
        self.fixed_array_pattern = re.compile(r"(.*)\[(\d+)\]")
        
        logger.info(f"LLMAgent initialized with RAG endpoint: {api_endpoint}")
    
    def fetch_rag_suggestion(self, prompt: str) -> Optional[str]:
        """
        Gọi API Flask để lấy gợi ý từ RAG với retry và fallback
        
        :param prompt: Câu hỏi hoặc prompt gửi tới RAG
        :return: Kết quả từ RAG hoặc None nếu lỗi
        """
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                response = requests.post(
                    self.api_endpoint,
                    json={"prompt": prompt},
                    timeout=30  # Giảm timeout để tránh treo
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"RAG raw response: {result}")
                    
                    if "response" in result:
                        return result["response"]
                    else:
                        logger.warning(f"RAG response missing 'response' field: {result}")
                else:
                    logger.warning(f"RAG server returned status code {response.status_code}")
                
            except requests.ConnectionError:
                logger.warning(f"Connection error to RAG server (retry {retry_count+1}/{max_retries+1})")
            except requests.Timeout:
                logger.warning(f"Timeout connecting to RAG server (retry {retry_count+1}/{max_retries+1})")
            except Exception as e:
                logger.warning(f"Error fetching RAG suggestion: {str(e)} (retry {retry_count+1}/{max_retries+1})")
            
            retry_count += 1
            if retry_count <= max_retries:
                # Đợi một chút trước khi thử lại (backoff)
                time.sleep(1 * retry_count)
        
        # Không thể kết nối đến RAG server, thử lấy giá trị từ prompt
        logger.warning("Failed to connect to RAG server, trying fallback from prompt")
        return self._get_fallback_from_prompt(prompt)
        
    def _get_fallback_from_prompt(self, prompt: str) -> Optional[str]:
        """
        Phân tích prompt để trả về giá trị mặc định phù hợp với kiểu dữ liệu
        
        :param prompt: Prompt gốc được gửi cho RAG
        :return: Giá trị mặc định phù hợp với kiểu
        """
        try:
            # Trích xuất kiểu dữ liệu từ prompt
            type_match = re.search(r"Parameter type: ([a-zA-Z0-9\[\]]+)", prompt)
            if not type_match:
                return None
                
            param_type = type_match.group(1)
            
            # Trả về giá trị mặc định dựa trên kiểu
            if param_type.startswith("uint"):
                return "0"  # Giá trị uint an toàn
            elif param_type.startswith("int"):
                return "0"  # Giá trị int an toàn
            elif param_type == "address":
                return "0x0000000000000000000000000000000000000000"  # zero address
            elif param_type == "bool":
                return "false"
            elif param_type.startswith("bytes"):
                return "0x00"
            elif param_type == "string":
                return ""
            else:
                return None
        except Exception as e:
            logger.error(f"Error in fallback generation: {e}")
            return None

    
    def get_argument_suggestion(self, 
                               type_str: str, 
                               function_name: str, 
                               arg_name: str, 
                               arg_index: int,
                               context: Optional[Dict] = None) -> Any:
        """
        Lấy đề xuất giá trị cho một tham số từ RAG
        
        :param type_str: Chuỗi kiểu dữ liệu (vd: "uint256", "address", "string")
        :param function_name: Tên hàm (vd: "constructor", "transfer")
        :param arg_name: Tên tham số (nếu có)
        :param arg_index: Vị trí tham số
        :param context: Bối cảnh bổ sung (audit report, phân tích...)
        :return: Giá trị được đề xuất (đã chuyển đổi sang kiểu thích hợp)
        """
        # Kiểm tra cache
        cache_key = f"{function_name}_{arg_name}_{type_str}_{arg_index}"
        logger.info(f"LLM AGENT: Generating suggestion for {cache_key}")
        if cache_key in self.cache:
            logger.info(f"LLM AGENT CACHE: Using cached value for {cache_key}: {self.cache[cache_key]}")
            return self.cache[cache_key]
        selector_fallback = None
        if function_name.startswith("0x"):  # Là function selector
            selector_fallback = self._get_default_for_function_selector(function_name, arg_index, type_str)
            if selector_fallback is not None:
                logger.info(f"Using selector-based fallback for {function_name}.{arg_name}: {selector_fallback}")
                return selector_fallback    
        # Xây dựng prompt cho RAG
        prompt = self._build_prompt(type_str, function_name, arg_name, arg_index, context)
        logger.debug(f"RAG PROMPT: {prompt}")
        
        # Gọi RAG qua API Flask
        rag_response = self.fetch_rag_suggestion(prompt)
        if rag_response:
            value = self._parse_rag_response(rag_response, type_str)
            typed_value = self._convert_to_type(value, type_str)
            if typed_value is not None:
                self.cache[cache_key] = typed_value
                logger.info(f"RAG SUCCESS: Generated value for {function_name}.{arg_name}: {typed_value}")
                return typed_value
        
        logger.warning(f"RAG FAILED: No valid value for {function_name}.{arg_name} ({type_str})")
        return None
    
    def _build_prompt(self, type_str: str, function_name: str, arg_name: str, arg_index: int, context: Optional[Dict]) -> str:
        prompt = f"""Base your smart contract audit reports, generate a SINGLE optimal value for fuzzing a Solidity smart contract function parameter based on the audit report. Return ONLY ONE value.

Parameter information:
- Function: {function_name}
- Parameter name: {arg_name}
- Parameter type: {type_str}
- Parameter index: {arg_index}

Goal: Generate a value that could potentially:
1. Trigger edge cases
2. Expose security vulnerabilities
3. Test boundary conditions
4. Exercise unusual code paths
"""

        # Thêm hướng dẫn dựa trên kiểu
        if type_str.startswith("uint"):
            prompt += """
For uint types:
- Return ONE positive integer (decimal or hex with '0x' prefix)
- Prioritize edge cases: 0, 1, 115792089237316195423570985008687907853269984665640564039457584007913129639935
- Examples: 0, 1, 1000, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
- Avoid lists, expressions (e.g., 2**256 - 1), or non-numeric values
"""
        elif type_str.startswith("int"):
            prompt += """
For int types:
- Return ONE integer (decimal or hex with '0x' prefix)
- Examples: -1, 0, 1000, 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
- Avoid lists or non-numeric values
"""
        elif type_str == "address":
            prompt += """
For address types:
- Return EXACTLY ONE valid 42-character Ethereum address with '0x' prefix
- Valid examples: 0x0000000000000000000000000000000000000000, 0xffffffffffffffffffffffffffffffffffffffff
- Special addresses to consider: address(0), address(this), address(1), contract deployer
- Format: 0x followed by exactly 40 hexadecimal digits (0-9, a-f)
- If you don't have specific addresses from the audit report, use one of these known important addresses:
* 0x0000000000000000000000000000000000000000 (zero address)
* 0x0000000000000000000000000000000000000001 (ecrecover precompile)
* 0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF (max address)
- RETURN ONLY THE ADDRESS with no additional text or explanation
"""
        elif type_str.startswith("bool"):
            prompt += """
For boolean types:
- Return ONE value: either "true" or "false"
- Avoid other values
"""
        elif type_str.startswith("bytes"):
            prompt += """
For bytes types:
- Return ONE hex string with '0x' prefix
- Examples: 0x, 0x1234, 0xFFFFFFFFFFFFFFFF
- Avoid non-hex values
"""
        elif type_str.startswith("string"):
            prompt += """
For string types:
- Return ONE string
- Examples: "", "test", "special chars: @#$"
- Avoid JSON or other formats
"""
            
        # Thêm bối cảnh audit nếu có
        if context and "audit_report" in context:
            prompt += f"""
Audit report context:
{context['audit_report']}

Based on this audit report:
1. Identify potential security issues
2. Generate a single value that could trigger these issues
3. Focus on edge cases mentioned in the report
4. Avoid returning lists or expressions
"""

        prompt += """
Return ONLY the raw value with NO explanation, NO JSON formatting, and NO quotes (unless it's a string).
The value MUST match the specified format for the type and be a SINGLE value.
"""
        return prompt
    
    def _parse_rag_response(self, response_text: str, type_str: str) -> Any:
        """
        Xử lý phản hồi từ RAG và trích xuất giá trị
        """
        # Làm sạch chuỗi phản hồi
        value = response_text.strip()
        
        # Xóa tiền tố "Response from AI:" nếu có
        if value.startswith("Response from AI:"):
            value = value[len("Response from AI:"):].strip()
        
        # Xử lý trường hợp "Không có thông tin phù hợp"
        if value == "Không có thông tin phù hợp":
            logger.warning(f"No relevant information found for type {type_str}")
            return None
        
        # Xóa dấu ngoặc kép nếu có
        if type_str != "string" and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        
        # Xóa các ký tự không mong muốn
        value = value.replace('\n', '').replace('\r', '').strip()
        
        # Kiểm tra giá trị rỗng
        if not value:
            logger.warning(f"Empty value received from RAG for type {type_str}")
            return None
        
        # Xử lý giá trị cho uint/int
        if type_str.startswith("uint") or type_str.startswith("int"):
            # Xử lý biểu thức như "2**256 - 1"
            if "2**256 - 1" in value:
                return "115792089237316195423570985008687907853269984665640564039457584007913129639935"
            
            # Xử lý danh sách số (như "1. 02. 13. ...")
            if "." in value:
                numbers = [n.strip() for n in value.split(".") if n.strip().isdigit()]
                if numbers:
                    return numbers[0]  # Lấy số đầu tiên hợp lệ
                logger.warning(f"Invalid numeric list received: {value}")
                return None
            
            # Chấp nhận decimal hoặc hex
            if value.replace("-", "").isdigit():
                return value
            elif value.startswith("0x"):
                try:
                    int(value, 16)  # Kiểm tra hex hợp lệ
                    return value
                except ValueError:
                    logger.warning(f"Invalid hex value received: {value}")
                    return None
            else:
                logger.warning(f"Invalid numeric value received: {value}")
                return None
        
        # Xử lý địa chỉ
        elif type_str == "address":
            # Tìm chuỗi giống địa chỉ Ethereum bằng regex
            address_pattern = re.search(r"0x[a-fA-F0-9]{40}", value)
            if address_pattern:
                return address_pattern.group(0)
                
            # Xử lý các alias phổ biến
            address_aliases = {
                "address(0)": "0x0000000000000000000000000000000000000000",
                "zero address": "0x0000000000000000000000000000000000000000",
                "address 0": "0x0000000000000000000000000000000000000000",
                "null address": "0x0000000000000000000000000000000000000000"
            }
            
            for alias, address in address_aliases.items():
                if alias in value.lower():
                    return address
                    
            # Fallback cho địa chỉ
            common_addresses = [
                "0x0000000000000000000000000000000000000000",  # zero address
                "0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF",  # max address
                "0x1111111111111111111111111111111111111111",  # một địa chỉ khác
                "0x2222222222222222222222222222222222222222"   # một địa chỉ khác
            ]
            
            # Fallback béo địa chỉ hữu ích cho fuzzing
            logger.info(f"Using fallback address for {type_str}")
            return random.choice(common_addresses)
        
        # Xử lý bytes
        elif type_str.startswith("bytes"):
            if not value.startswith("0x"):
                value = "0x" + value
            try:
                int(value[2:], 16)  # Kiểm tra hex hợp lệ
                return value
            except ValueError:
                logger.warning(f"Invalid bytes format: {value}")
                return None
        
        # Trả về giá trị thô cho các kiểu khác
        return value

    def report_vulnerability_to_rag(self, 
                            transaction_id: str, 
                            function_name: str, 
                            vulnerability_type: str,
                            args: List[Any],
                            source: str = "rag",
                            description: str = "") -> bool:
        """
        Báo cáo lỗi về server RAG để theo dõi hiệu quả
        
        :param transaction_id: ID của transaction phát hiện lỗi
        :param function_name: Tên hàm gây ra lỗi
        :param vulnerability_type: Loại lỗi (overflow, reentrancy, etc.)
        :param args: Các tham số của hàm gây lỗi
        :param source: Nguồn tạo ra giá trị (rag/random)
        :param description: Mô tả chi tiết về lỗi
        :return: True nếu báo cáo thành công
        """
        try:
            payload = {
                "transaction_id": transaction_id,
                "function_name": function_name,
                "vulnerability_type": vulnerability_type,
                "args": args,
                "source": source,
                "description": description
            }
            
            # Extract host and port from API endpoint
            url_parts = self.api_endpoint.split("/")
            base_url = "/".join(url_parts[:-1])  # Remove last part ("request")
            report_url = f"{base_url}/report_vulnerability"
            
            response = requests.post(
                report_url,
                json=payload,
                timeout=5  # Short timeout for reporting
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully reported vulnerability: {vulnerability_type} in {function_name}")
                return True
            else:
                logger.warning(f"Failed to report vulnerability. Status code: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Error reporting vulnerability to RAG server: {e}")
            return False
    
    def _get_default_for_function_selector(self, selector: str, arg_index: int, type_str: str) -> Any:
        """
        Trả về giá trị mặc định cho tham số dựa trên function selector
        """
        # Tạo các giá trị đặc biệt có khả năng gây lỗi tràn số
        UINT_MAX = (2**256) - 1
        UINT_MAX_MINUS_1 = UINT_MAX - 1
        UINT_LARGE = 2**255
        UINT_HALF = 2**128
        KNOWN_OVERFLOW_VALUES = [
            UINT_MAX, 
            UINT_MAX_MINUS_1,
            UINT_LARGE,
            UINT_HALF,
            UINT_MAX - 10,
            2**250,
            2**200,
            2**100 + 1
        ]
        
        # balanceOf(address)
        if selector == "0x70a08231" and arg_index == 0 and type_str == "address":
            special_addresses = [
                "0x0000000000000000000000000000000000000000",  # zero address
                "0x0000000000000000000000000000000000000001",  # precompile
                "0x000000000000000000000000000000000000dEaD",  # burn address
            ]
            return random.choice(special_addresses)
        
        # transfer(address,uint256)
        if selector == "0xa9059cbb":
            if arg_index == 0 and type_str == "address":
                return "0x0000000000000000000000000000000000000001"  # địa chỉ nhận
            elif arg_index == 1 and type_str.startswith("uint"):
                # Chọn ngẫu nhiên giữa các giá trị có khả năng gây overflow
                return random.choice(KNOWN_OVERFLOW_VALUES)
        
        # approve(address,uint256)
        if selector == "0x095ea7b3":
            if arg_index == 0 and type_str == "address":
                return "0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF"  # spender
            elif arg_index == 1 and type_str.startswith("uint"):
                return UINT_MAX  # Giá trị tối đa
        
        # transferFrom(address,address,uint256)
        if selector == "0x23b872dd":
            if arg_index == 0 and type_str == "address":
                return "0x0000000000000000000000000000000000000000"  # from (owner)
            elif arg_index == 1 and type_str == "address":  # arg1 - địa chỉ đang gây lỗi
                return "0x0000000000000000000000000000000000000001"  # to (recipient)
            elif arg_index == 2 and type_str.startswith("uint"):
                return random.choice([UINT_MAX, UINT_LARGE, 1000000])
        
        # decreaseAllowance(address,uint256)
        if selector == "0xa457c2d7":
            if arg_index == 0 and type_str == "address":
                return "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
            elif arg_index == 1 and type_str.startswith("uint"):
                # Thử gọi decrease với giá trị lớn hơn allowance
                return UINT_MAX_MINUS_1
        
        # increaseAllowance(address,uint256)
        if selector == "0x39509351":
            if arg_index == 0 and type_str == "address":
                return "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
            elif arg_index == 1 and type_str.startswith("uint"):
                # Thử tăng allowance với giá trị gây overflow
                return random.choice([2, UINT_MAX, UINT_LARGE])
        
        # Fallback: các giá trị thông thường theo kiểu
        if type_str.startswith("uint") or type_str.startswith("int"):
            # Có 20% xác suất trả về giá trị đặc biệt gây overflow
            if random.random() < 0.2:
                return random.choice(KNOWN_OVERFLOW_VALUES)
            # 10% xác suất sẽ trả về 0
            elif random.random() < 0.1:
                return 0
            # 10% xác suất trả về giá trị âm (cho int)
            elif random.random() < 0.1 and type_str.startswith("int"):
                return -1
            # 60% còn lại trả về giá trị ngẫu nhiên trong phạm vi hợp lý
            else:
                return random.randint(1, 1000000)
                
        return None

    
    def _convert_to_type(self, value: str, type_str: str) -> Any:
        """
        Chuyển đổi giá trị chuỗi sang kiểu dữ liệu thích hợp
        """
        if value is None:
            return None
            
        logger.debug(f"Converting value: {value} for type: {type_str}")
        
        # Mảng
        array_match = self.array_pattern.match(type_str)
        fixed_array_match = self.fixed_array_pattern.match(type_str)
        
        if array_match or fixed_array_match:
            if array_match:
                base_type = array_match.group(0)
                # Trường hợp mảng động, bắt đầu với một phần tử
                return [self._convert_to_type(value, base_type)]
            else:
                base_type = fixed_array_match.group(1)
                size = int(fixed_array_match.group(2))
                # Tạo mảng với kích thước cố định, tất cả có cùng giá trị
                return [self._convert_to_type(value, base_type) for _ in range(size)]
        
        # Boolean
        if type_str == "bool":
            return value.lower() in ["true", "1", "yes"]
        
        # Unsigned integer
        uint_match = self.uint_pattern.match(type_str)
        if uint_match:
            try:
                num = int(value)
                # Kiểm tra giới hạn uint
                bits = int(uint_match.group(1))
                max_value = (1 << bits) - 1
                if num < 0 or num > max_value:
                    logger.warning(f"Value {num} out of range for uint{bits}")
                    return 0
                return num
            except ValueError:
                # Xử lý giá trị hex
                if value.startswith("0x"):
                    try:
                        return int(value, 16)
                    except ValueError:
                        logger.warning(f"Invalid hex value: {value}")
                        return 0
                return 0
        
        # Signed integer
        int_match = self.int_pattern.match(type_str)
        if int_match:
            try:
                num = int(value)
                # Kiểm tra giới hạn int
                bits = int(int_match.group(1))
                max_value = (1 << (bits - 1)) - 1
                min_value = -(1 << (bits - 1))
                if num < min_value or num > max_value:
                    logger.warning(f"Value {num} out of range for int{bits}")
                    return 0
                return num
            except ValueError:
                # Xử lý giá trị hex
                if value.startswith("0x"):
                    try:
                        return int(value, 16)
                    except ValueError:
                        logger.warning(f"Invalid hex value: {value}")
                        return 0
                return 0
        
        # Address
        if type_str == "address":
            # Đảm bảo địa chỉ hợp lệ
            if not value.startswith("0x"):
                value = "0x" + value
            if len(value) != 42:
                # Pad cho đủ 42 ký tự
                value = "0x" + value[2:].zfill(40)
            return value
        
        # String
        if type_str == "string":
            return value
        
        # Bytes và bytesN
        bytes_match = self.bytes_pattern.match(type_str)
        if type_str == "bytes" or bytes_match:
            if value.startswith("0x"):
                try:
                    return bytes.fromhex(value[2:])
                except ValueError:
                    logger.warning(f"Invalid hex bytes: {value}")
                    return b""
            else:
                return bytes(value, 'utf-8')
        
        # Mặc định trả về chuỗi
        return value