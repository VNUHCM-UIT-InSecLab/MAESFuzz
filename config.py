import loguru
import os

loguru.logger.add("log/DEBUG.log",   encoding="utf-8", enqueue=True, backtrace=True, diagnose=True, level="DEBUG")
loguru.logger.add("log/INFO.log",    encoding="utf-8", enqueue=True, backtrace=True, diagnose=True, level="INFO")
loguru.logger.add("log/ERROR.log",   encoding="utf-8", enqueue=True, backtrace=True, diagnose=True, level="ERROR")
loguru.logger.add("log/WARNING.log", encoding="utf-8", enqueue=True, backtrace=True, diagnose=True, level="WARNING")

# Solidity compiler
SOLC_BIN_PATH       = "/Users/coolstar/.solc-select/artifacts/solc-0.4.26/solc-0.4.26"
DEFAULT_SOLC_VERSION = "0.8.26"
DEFAULT_EVM_VERSION  = "byzantium"

# Fuzzing
DEFAULT_FUZZ_TIME        = 60
DEFAULT_MAX_TRANS_LENGTH = 10
DEFAULT_DUPLICATION      = "0"

# Output
DEFAULT_RESULT_PATH       = "result/results.json"
DEFAULT_CONSTRUCTOR_PARAMS = "auto"

# RAG / LLM
DEFAULT_USE_RAG       = True
DEFAULT_LLM_MODEL     = os.environ.get("LLM_MODEL",    "gpt-4o")
DEFAULT_LLM_PROVIDER  = os.environ.get("LLM_PROVIDER", "openai")
DEFAULT_OLLAMA_ENDPOINT  = os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_THINKING  = os.environ.get("OLLAMA_THINKING")

# LLM-as-a-Judge
DEFAULT_USE_LLM_JUDGE  = os.environ.get("USE_LLM_JUDGE", "false").lower() == "true"
DEFAULT_JUDGE_THRESHOLD = float(os.environ.get("JUDGE_THRESHOLD", "60.0"))

# ContraMaster
ENABLE_CONTRAMASTER            = True
ENABLE_SEMANTIC_ORACLE         = True
ENABLE_BOOKKEEPING_AUTO_DETECT = True
ENABLE_INTERVAL_GAS_MUTATION   = True
GAS_MUTATION_INTERVALS         = 10
ENABLE_DATA_DRIVEN_SEQUENCE    = True
SEQUENCE_MUTATION_PROBABILITY  = 0.3
BOOKKEEPING_TEST_TRANSACTIONS  = 10
BOOKKEEPING_SUCCESS_THRESHOLD  = 0.8


def get_logger() -> loguru.logger:  # type: ignore
    return loguru.logger


def get_solc_path() -> str:
    return SOLC_BIN_PATH


def get_default_solc_version() -> str:
    return DEFAULT_SOLC_VERSION


def get_default_fuzz_time() -> int:
    return DEFAULT_FUZZ_TIME


def get_default_max_trans_length() -> int:
    return DEFAULT_MAX_TRANS_LENGTH


def get_default_duplication() -> str:
    return DEFAULT_DUPLICATION


def get_default_result_path() -> str:
    return DEFAULT_RESULT_PATH


def get_default_constructor_params() -> str:
    return DEFAULT_CONSTRUCTOR_PARAMS


def get_default_evm_version() -> str:
    return DEFAULT_EVM_VERSION


def get_evm_version_for_solc(solc_version: str) -> str:
    """Return the appropriate EVM version for a given Solidity compiler version.

    UniFuzz supports: 'homestead', 'byzantium', 'petersburg'.
    Newer EVM targets (istanbul, berlin, london …) fall back to 'petersburg'.
    """
    try:
        version_str = str(solc_version).strip().lstrip("v")
        parts = version_str.split(".")
        if len(parts) < 2:
            return DEFAULT_EVM_VERSION
        major, minor = int(parts[0]), int(parts[1])
        if major == 0:
            if minor < 5:
                return "byzantium"
            return "petersburg"
        return "petersburg"
    except (ValueError, IndexError):
        return DEFAULT_EVM_VERSION


def is_contramaster_enabled() -> bool:
    return ENABLE_CONTRAMASTER


def is_semantic_oracle_enabled() -> bool:
    return ENABLE_CONTRAMASTER and ENABLE_SEMANTIC_ORACLE


def is_interval_gas_enabled() -> bool:
    return ENABLE_CONTRAMASTER and ENABLE_INTERVAL_GAS_MUTATION


def is_sequence_mutation_enabled() -> bool:
    return ENABLE_CONTRAMASTER and ENABLE_DATA_DRIVEN_SEQUENCE


def get_gas_mutation_intervals() -> int:
    return GAS_MUTATION_INTERVALS


def get_sequence_mutation_probability() -> float:
    return SEQUENCE_MUTATION_PROBABILITY


def get_default_use_rag() -> bool:
    return DEFAULT_USE_RAG


def get_google_api_key() -> str:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key
    try:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOOGLE_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
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
    return DEFAULT_USE_LLM_JUDGE


def get_default_judge_threshold() -> float:
    return DEFAULT_JUDGE_THRESHOLD