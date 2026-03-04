import loguru
import os

loguru.logger.add("log/DEBUG.log", encoding="utf-8", enqueue=True, backtrace=True, diagnose=True, level="DEBUG")
loguru.logger.add("log/INFO.log", encoding="utf-8", enqueue=True, backtrace=True, diagnose=True, level="INFO")
loguru.logger.add("log/ERROR.log", encoding="utf-8", enqueue=True, backtrace=True, diagnose=True, level="ERROR")
loguru.logger.add("log/WARNING.log", encoding="utf-8", enqueue=True, backtrace=True, diagnose=True, level="WARNING")

# ==================== FUZZING CONFIGURATION ====================
# Solidity Compiler Configuration
SOLC_BIN_PATH = "/Users/coolstar/.solc-select/artifacts/solc-0.4.26/solc-0.4.26"
#DEFAULT_SOLC_VERSION = "0.4.26
DEFAULT_SOLC_VERSION = "0.8.26"  # Default Solidity version

# Fuzzing Configuration
DEFAULT_FUZZ_TIME = 60  # Default fuzzing time in seconds
DEFAULT_MAX_TRANS_LENGTH = 10  # Default max transaction sequence length
DEFAULT_DUPLICATION = "0"  # Default duplication mode (0=no duplicates, 1=allow duplicates)

# Output Configuration
DEFAULT_RESULT_PATH = "result/results.json"  # Default results output path
DEFAULT_CONSTRUCTOR_PARAMS = "auto"  # Default constructor params mode

# EVM Configuration
DEFAULT_EVM_VERSION = "byzantium"

# ==================== RAG & LLM CONFIGURATION ====================
# RAG Configuration
DEFAULT_USE_RAG = True  # Enable RAG by default
DEFAULT_GOOGLE_API_KEY = None  # Will be loaded from environment or .env file

# LLM Provider Configuration
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")
DEFAULT_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
DEFAULT_OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_THINKING = os.environ.get("OLLAMA_THINKING")

# LLM-as-a-Judge Configuration
DEFAULT_USE_LLM_JUDGE = os.environ.get("USE_LLM_JUDGE", "false").lower() == "true"
DEFAULT_JUDGE_THRESHOLD = float(os.environ.get("JUDGE_THRESHOLD", "60.0"))

# ==================== CONTRAMASTER INTEGRATION ====================
# Enable ContraMaster enhancements (Wang et al. 2020)
ENABLE_CONTRAMASTER = True

# Semantic Test Oracle Configuration
ENABLE_SEMANTIC_ORACLE = True  # Enable Balance & Transaction Invariants
ENABLE_BOOKKEEPING_AUTO_DETECT = True  # Auto-identify bookkeeping variables

# Gas Mutation Configuration
ENABLE_INTERVAL_GAS_MUTATION = True  # Enable interval-based gas mutation
GAS_MUTATION_INTERVALS = 10  # Number of intervals to divide gas range

# Sequence Mutation Configuration
ENABLE_DATA_DRIVEN_SEQUENCE = True  # Enable data-dependency-driven sequence mutation
SEQUENCE_MUTATION_PROBABILITY = 0.3  # Probability of applying sequence mutation

# Bookkeeping Identification Configuration
BOOKKEEPING_TEST_TRANSACTIONS = 10  # Number of test transactions for identification
BOOKKEEPING_SUCCESS_THRESHOLD = 0.8  # Success rate threshold (80%)

# ==================== HELPER FUNCTIONS ====================
def get_logger() -> loguru.logger: # type: ignore
    return loguru.logger

def get_solc_path() -> str:
    """Get the Solidity compiler path"""
    return SOLC_BIN_PATH

def get_default_solc_version() -> str:
    """Get the default Solidity version"""
    return DEFAULT_SOLC_VERSION

def get_default_fuzz_time() -> int:
    """Get the default fuzzing time"""
    return DEFAULT_FUZZ_TIME

def get_default_max_trans_length() -> int:
    """Get the default max transaction length"""
    return DEFAULT_MAX_TRANS_LENGTH

def get_default_duplication() -> str:
    """Get the default duplication mode"""
    return DEFAULT_DUPLICATION

def get_default_result_path() -> str:
    """Get the default result path"""
    return DEFAULT_RESULT_PATH

def get_default_constructor_params() -> str:
    """Get the default constructor params mode"""
    return DEFAULT_CONSTRUCTOR_PARAMS

def get_default_evm_version() -> str:
    """Get the default EVM version"""
    return DEFAULT_EVM_VERSION

def get_evm_version_for_solc(solc_version: str) -> str:
    """
    Automatically select appropriate EVM version based on Solidity compiler version.
    
    Mapping based on Solidity release notes and changelog:
    Reference: https://ethereum.stackexchange.com/questions/159166/how-to-know-the-default-evm-version-of-a-specific-version-of-solc
    
    Default EVM versions by Solidity version:
    - 0.4.21+: byzantium (default from 0.4.21)
    - 0.5.5+: petersburg (default from 0.5.5)
    - 0.5.14: istanbul (default, but UniFuzz doesn't support, fallback to petersburg)
    - 0.8.5: berlin (default, but UniFuzz doesn't support, fallback to petersburg)
    - 0.8.7: london (default, but UniFuzz doesn't support, fallback to petersburg)
    - 0.8.18: paris (default, but UniFuzz doesn't support, fallback to petersburg)
    - 0.8.20: shanghai (default, but UniFuzz doesn't support, fallback to petersburg)
    - 0.8.25+: cancun (default, but UniFuzz doesn't support, fallback to petersburg)
    
    Note: UniFuzz only supports: 'homestead', 'byzantium', 'petersburg'
    For unsupported EVM versions, we fallback to the most recent supported version.
    
    Args:
        solc_version: Solidity compiler version string (e.g., "0.4.26", "0.5.8", "0.8.23")
    
    Returns:
        EVM version string: "byzantium" or "petersburg" (only supported by UniFuzz)
    """
    try:
        version_str = str(solc_version).strip()
        if version_str.startswith("v"):
            version_str = version_str[1:]
        
        parts = version_str.split(".")
        if len(parts) < 2:
            return DEFAULT_EVM_VERSION
        
        major = int(parts[0])
        minor = int(parts[1])
        
        if major == 0:
            # Solidity 0.4.x: byzantium (default from 0.4.21)
            if minor < 5:
                return "byzantium"
            # Solidity 0.5.x
            elif minor == 5:
                # 0.5.5 - 0.5.13: petersburg (default from 0.5.5)
                # 0.5.14: istanbul (default, but UniFuzz doesn't support, fallback to petersburg)
                return "petersburg"
            # Solidity 0.6.x, 0.7.x, 0.8.x: newer EVM versions, fallback to petersburg
            elif minor >= 6:
                # 0.6.x, 0.7.x, 0.8.x use newer EVM versions (istanbul, berlin, london, paris, shanghai, cancun)
                # but UniFuzz only supports petersburg, so fallback
                return "petersburg"
            else:
                return "byzantium"
        elif major >= 1:
            # Future Solidity 1.x.x versions: use petersburg (most recent supported by UniFuzz)
            return "petersburg"
        else:
            return DEFAULT_EVM_VERSION
    except (ValueError, IndexError):
        return DEFAULT_EVM_VERSION

# ContraMaster Configuration Helpers
def is_contramaster_enabled() -> bool:
    """Check if ContraMaster enhancements are enabled"""
    return ENABLE_CONTRAMASTER

def is_semantic_oracle_enabled() -> bool:
    """Check if semantic oracle is enabled"""
    return ENABLE_CONTRAMASTER and ENABLE_SEMANTIC_ORACLE

def is_interval_gas_enabled() -> bool:
    """Check if interval-based gas mutation is enabled"""
    return ENABLE_CONTRAMASTER and ENABLE_INTERVAL_GAS_MUTATION

def is_sequence_mutation_enabled() -> bool:
    """Check if data-driven sequence mutation is enabled"""
    return ENABLE_CONTRAMASTER and ENABLE_DATA_DRIVEN_SEQUENCE

def get_gas_mutation_intervals() -> int:
    """Get number of intervals for gas mutation"""
    return GAS_MUTATION_INTERVALS

def get_sequence_mutation_probability() -> float:
    """Get probability of applying sequence mutation"""
    return SEQUENCE_MUTATION_PROBABILITY

# RAG Configuration Helpers
def get_default_use_rag() -> bool:
    """Get default RAG usage setting"""
    return DEFAULT_USE_RAG

def get_google_api_key() -> str:
    """Get Google API key from environment or .env file"""
    # Try environment variable first
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key
    
    # Try loading from .env file
    try:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOOGLE_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        return api_key
    except Exception:
        pass
    
    return None

def get_default_llm_model() -> str:
    return DEFAULT_LLM_MODEL

def get_default_llm_provider() -> str:
    return DEFAULT_LLM_PROVIDER

def get_default_ollama_endpoint() -> str:
    return DEFAULT_OLLAMA_ENDPOINT

def get_default_ollama_thinking():
    return DEFAULT_OLLAMA_THINKING

def get_default_use_llm_judge() -> bool:
    """Get default LLM-as-a-Judge usage setting"""
    return DEFAULT_USE_LLM_JUDGE

def get_default_judge_threshold() -> float:
    """Get default LLM Judge quality threshold (0-100)"""
    return DEFAULT_JUDGE_THRESHOLD