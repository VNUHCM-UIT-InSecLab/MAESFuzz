import json
import shutil
import sys
import os
import logging

import config
from config import (
    get_solc_path, get_default_solc_version, get_default_fuzz_time,
    get_default_max_trans_length, get_default_duplication, 
    get_default_result_path, get_default_constructor_params, get_default_evm_version,
    get_evm_version_for_solc,
    get_default_use_rag, get_google_api_key, get_default_llm_model,
    get_default_llm_provider, get_default_ollama_endpoint, get_default_ollama_thinking,
    get_default_use_llm_judge, get_default_judge_threshold
)
from fuzzer.utils.comp import analysis_depend_contract
from fuzzer.engine.analyzers.Analysis import analysis_main_contract_constructor

logger = logging.getLogger(__name__)


def run(_file_path: str, _main_contract, solc_version: str, evm_version: str, timeout: int, _depend_contracts: list,
        max_individual_length: int, _constructor_args: list, _solc_path: str, _duplication: str = '0',
        generations: int = None,
        use_rag: bool = True, api_key: str = None, llm_model: str = None, llm_provider: str = None,
        ollama_endpoint: str = None, ollama_thinking: str = None,
        openai_api_key: str = None,
        use_llm_judge: bool = False, judge_threshold: float = 60.0):
    # Log các tham số đầu vào
    logger.info("=== Thông tin tham số đầu vào ===")
    logger.info(f"File hợp đồng: {_file_path}")
    logger.info(f"Hợp đồng chính: {_main_contract}")
    logger.info(f"Phiên bản Solc: {solc_version}")
    logger.info(f"Phiên bản EVM: {evm_version}")
    logger.info(f"Thời gian timeout: {timeout}")
    if generations:
        logger.info(f"Số thế hệ: {generations}")
    logger.info(f"Các hợp đồng phụ thuộc: {_depend_contracts}")
    logger.info(f"Độ dài tối đa của cá thể: {max_individual_length}")
    logger.info(f"Tham số constructor: {_constructor_args}")
    logger.info(f"Đường dẫn Solc: {_solc_path}")
    logger.info(f"Chế độ trùng lặp: {_duplication}")
    logger.info(f"LLM Model: {llm_model}")
    logger.info(f"LLM Provider: {llm_provider or 'auto'}")
    if ollama_endpoint:
        logger.info(f"Ollama endpoint: {ollama_endpoint}")
    if ollama_thinking:
        logger.info(f"Ollama thinking: {ollama_thinking}")
    logger.info(f"LLM-as-a-Judge: {'Enabled' if use_llm_judge else 'Disabled'}")
    if use_llm_judge:
        logger.info(f"Judge Threshold: {judge_threshold}")
    logger.info("===============================")

    depend_contracts_str = " ".join(_depend_contracts) if _depend_contracts else ""
    
    # Convert constructor args to strings if they are tuples or dicts
    constructor_args_strs = []
    for arg in _constructor_args:
        if isinstance(arg, (tuple, list)) and len(arg) >= 3:
            # Format: (name, type, value, ...) -> "name type value"
            constructor_args_strs.append(f"{arg[0]} {arg[1]} {arg[2]}")
        elif isinstance(arg, dict):
            # Format: {"name": ..., "type": ..., "value": ...} -> "name type value"
            constructor_args_strs.append(f"{arg.get('name', '')} {arg.get('type', '')} {arg.get('value', '')}")
        elif isinstance(arg, str):
            constructor_args_strs.append(arg)
        else:
            constructor_args_strs.append(str(arg))
    constructor_str = " ".join(constructor_args_strs) if constructor_args_strs else ""
    # Build command incrementally to allow omitting timeout when generations is set
    cmd_parts = [
        f"{PYTHON} {FUZZER}",
        f"-s {_file_path}",
        f"-c {_main_contract}",
        f"--solc v{solc_version}",
        f"--evm {evm_version}",
    ]
    if generations:
        cmd_parts.append(f"--generations {generations}")
    else:
        cmd_parts.append(f"-t {timeout}")
    cmd_parts.extend([
        "--result result/res.json",
        "--cross-contract 1",
        "--open-trans-comp 1",
        f"--depend-contracts {depend_contracts_str}",
        f"--constructor-args {constructor_str}",
        "--constraint-solving 1",
        f"--max-individual-length {max_individual_length}",
        f"--solc-path-cross {_solc_path}",
        "--p-open-cross 80",
        "--cross-init-mode 1",
        "--trans-mode 1",
        f"--duplication {_duplication}",
    ])
    
    # Add RAG parameters
    if use_rag:
        cmd_parts.append("--use-rag")
    if api_key:
        cmd_parts.append(f"--api-key {api_key}")
    if openai_api_key:
        cmd_parts.append(f"--openai-api-key {openai_api_key}")
    if llm_model:
        cmd_parts.append(f"--model {llm_model}")
    if llm_provider:
        cmd_parts.append(f"--provider {llm_provider}")
    if ollama_endpoint:
        cmd_parts.append(f"--ollama-endpoint {ollama_endpoint}")
    if ollama_thinking:
        cmd_parts.append(f"--ollama-thinking {ollama_thinking}")

    # Add LLM-as-a-Judge parameters
    if use_llm_judge:
        cmd_parts.append("--use-llm-judge")
        cmd_parts.append(f"--judge-threshold {judge_threshold}")

    # Pass through plateau controls if provided to UniFuzz CLI
    if "--no-plateau" in sys.argv:
        cmd_parts.append("--no-plateau")
    if "--plateau-generations" in sys.argv:
        idx = sys.argv.index("--plateau-generations")
        if idx + 1 < len(sys.argv):
            cmd_parts.append(f"--plateau-generations {sys.argv[idx+1]}")
    if "--plateau-threshold" in sys.argv:
        idx = sys.argv.index("--plateau-threshold")
        if idx + 1 < len(sys.argv):
            cmd_parts.append(f"--plateau-threshold {sys.argv[idx+1]}")

    cmd = " ".join(cmd_parts)

    # Print a redacted command to avoid leaking API keys
    redacted_cmd = cmd.replace(api_key, "***API***") if api_key else cmd
    if openai_api_key:
        redacted_cmd = redacted_cmd.replace(openai_api_key, "***OPENAI_API***")
    print(redacted_cmd)

    os.popen(cmd).readlines()  # run fuzzer/main.py

    # Thêm trong fuzzer/main.py sau khi gọi analysis_depend_contract
    logger.info("Kết quả phân tích phụ thuộc:")
    logger.info(f"Các hợp đồng phụ thuộc: {_depend_contracts}")

    # Thêm trong fuzzer/main.py sau khi gọi analysis_main_contract_constructor
    logger.info("Kết quả phân tích constructor:")
    logger.info(f"Các tham số constructor: {_constructor_args}")

    return "result/res.json"


def test_run():
    # absolute path
    _file_path = "./examples/reentrance.sol"
    _main_contract = "Reentrance"
    solc_version = "0.4.26"
    evm_version = get_evm_version_for_solc(solc_version)
    timeout = 10
    solc_path = config.SOLC_BIN_PATH
    _depend_contracts, _sl = analysis_depend_contract(file_path=_file_path, _contract_name=_main_contract,
                                                      _solc_version=solc_version, _solc_path=solc_path)
    max_individual_length = 10
    _constructor_args = analysis_main_contract_constructor(file_path=_file_path, _contract_name=_main_contract, sl=_sl)
    
    # Log thông tin phân tích
    logger.info("=== Kết quả phân tích ===")
    logger.info(f"Hợp đồng phụ thuộc: {_depend_contracts}")
    logger.info(f"Tham số constructor: {_constructor_args}")
    logger.info("========================")
    
    run(_file_path, _main_contract, solc_version, evm_version, timeout, _depend_contracts, max_individual_length,
        _constructor_args, _solc_path=config.SOLC_BIN_PATH)


def cli():
    if len(sys.argv) < 3:
        print("Usage: python MAESFuzz.py <sol_file_path> <contract_name> [optional_params]")
        print("")
        print("Required parameters:")
        print("  sol_file_path    : Path to the Solidity contract file")
        print("  contract_name    : Name of the contract to fuzz")
        print("")
        print("Optional parameters (use config defaults if not specified):")
        print("  --solc-version   : Solidity version (default: 0.4.26)")
        print("  --max-trans-length: Max transaction sequence length (default: 10)")
        print("  --fuzz-time      : Fuzzing time in seconds (default: 60)")
        print("  -g, --generations: Number of generations (overrides time stop if set)")
        print("  --result-path    : Output path for results (default: result/results.json)")
        print("  --constructor-params: Constructor params mode (default: auto)")
        print("  --duplication    : Allow duplicate transactions (default: 0)")
        print("  --no-rag         : Disable RAG-enhanced fuzzing (RAG is enabled by default)")
        print("  --api-key        : Google API Key for Gemini (or set GOOGLE_API_KEY env)")
        print("  --openai-api-key : OpenAI API Key for GPT models (or set OPENAI_API_KEY env)")
        print("  --model          : LLM model identifier (default: config.LLM_MODEL or gpt-4o)")
        print("  --provider       : LLM provider (auto | gemini | ollama | openai)")
        print("  --ollama-endpoint: Ollama endpoint (default: http://127.0.0.1:11434)")
        print("  --ollama-thinking: Enable thinking trace for Ollama models (true/false/low/high/medium)")
        print("  --use-llm-judge  : Enable LLM-as-a-Judge for mutation/crossover quality evaluation")
        print("  --judge-threshold: Quality score threshold for LLM Judge (0-100, default: 60)")
        print("")
        print("Examples:")
        print("  ./venv/bin/python MAESFuzz.py examples/reentrance.sol Reentrance")
        print("  ./venv/bin/python MAESFuzz.py examples/reentrance.sol Reentrance --fuzz-time 120 --max-trans-length 5")
        sys.exit(1)
    
    # Required parameters
    p = sys.argv[1]  # sol file path
    c_name = sys.argv[2]  # contract name
    
    # Optional parameters with defaults from config
    solc_version = get_default_solc_version()
    max_trans_length = get_default_max_trans_length()
    fuzz_time = get_default_fuzz_time()
    generations = None
    res_saved_path = get_default_result_path()
    solc_path = get_solc_path()
    constructor_params_path = get_default_constructor_params()
    trans_duplication = get_default_duplication()
    use_rag = get_default_use_rag()  # RAG enabled by default
    api_key = get_google_api_key()  # Get from env or .env
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    llm_model = get_default_llm_model()
    llm_provider = get_default_llm_provider()
    ollama_endpoint = get_default_ollama_endpoint()
    ollama_thinking = get_default_ollama_thinking()
    use_llm_judge = get_default_use_llm_judge()  # LLM-as-a-Judge
    judge_threshold = get_default_judge_threshold()
    
    # Parse optional arguments
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--solc-version" and i + 1 < len(sys.argv):
            solc_version = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--max-trans-length" and i + 1 < len(sys.argv):
            max_trans_length = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--fuzz-time" and i + 1 < len(sys.argv):
            fuzz_time = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] in ("-g", "--generations") and i + 1 < len(sys.argv):
            generations = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--result-path" and i + 1 < len(sys.argv):
            res_saved_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--constructor-params" and i + 1 < len(sys.argv):
            constructor_params_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--duplication" and i + 1 < len(sys.argv):
            trans_duplication = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--no-rag":
            use_rag = False
            i += 1
        elif sys.argv[i] == "--api-key" and i + 1 < len(sys.argv):
            api_key = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--openai-api-key" and i + 1 < len(sys.argv):
            openai_api_key = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            llm_model = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--provider" and i + 1 < len(sys.argv):
            llm_provider = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--ollama-endpoint" and i + 1 < len(sys.argv):
            ollama_endpoint = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--ollama-thinking" and i + 1 < len(sys.argv):
            ollama_thinking = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--use-llm-judge":
            use_llm_judge = True
            i += 1
        elif sys.argv[i] == "--judge-threshold" and i + 1 < len(sys.argv):
            judge_threshold = float(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    _depend_contracts, _sl = analysis_depend_contract(file_path=p, _contract_name=c_name, _solc_version=solc_version,
                                                      _solc_path=solc_path)
    if len(_depend_contracts) <= 0:
        print("Không có hợp đồng phụ thuộc - chạy với hợp đồng đơn lẻ")
        _depend_contracts = []  # Cho phép chạy với hợp đồng đơn lẻ
    if constructor_params_path != "auto":
        _constructor_args = []
        for p_name, p_detail in json.load(open(constructor_params_path, "r", encoding="utf-8")).items():
            _constructor_args.append(f"{p_name} {p_detail['type']} {p_detail['value']}")
    else:
        _constructor_args = analysis_main_contract_constructor(file_path=p, contract_name=c_name, sl=_sl)
    if _constructor_args is None:
        print("Không có constructor")
        sys.exit(-2)
    
    # Log thông tin cấu hình và phân tích
    logger.info("=== Cấu hình Fuzzing ===")
    logger.info(f"File hợp đồng: {p}")
    logger.info(f"Tên hợp đồng: {c_name}")
    logger.info(f"Phiên bản Solc: {solc_version}")
    logger.info(f"Thời gian fuzzing: {fuzz_time}s")
    if generations:
        logger.info(f"Số thế hệ: {generations}")
    logger.info(f"Độ dài tối đa sequence: {max_trans_length}")
    logger.info(f"Đường dẫn kết quả: {res_saved_path}")
    logger.info(f"Chế độ trùng lặp: {trans_duplication}")
    logger.info(f"Sử dụng RAG: {use_rag}")
    if api_key:
        logger.info(f"API Key: {'***' + api_key[-4:] if len(api_key) >= 4 else '***'}")
    else:
        logger.info("API Key: Không có")
    logger.info(f"LLM Provider: {llm_provider or 'auto'}")
    logger.info(f"LLM Model: {llm_model}")
    if ollama_endpoint:
        logger.info(f"Ollama endpoint: {ollama_endpoint}")
    if ollama_thinking:
        logger.info(f"Ollama thinking: {ollama_thinking}")
    logger.info(f"LLM-as-a-Judge: {'Enabled' if use_llm_judge else 'Disabled'}")
    if use_llm_judge:
        logger.info(f"Judge Threshold: {judge_threshold}")
    logger.info("========================")
    logger.info("=== Kết quả phân tích ===")
    logger.info(f"Hợp đồng phụ thuộc: {_depend_contracts}")
    logger.info(f"Tham số constructor: {_constructor_args}")
    logger.info("========================")
    
    evm_version = get_evm_version_for_solc(solc_version)
    logger.info(f"Auto-selected EVM version: {evm_version} (for Solidity {solc_version})")
    logger.info(f"Note: User must set solc version manually with 'solc-select use <version>' before running")
    res = run(p, c_name, solc_version, evm_version,
              fuzz_time, _depend_contracts, max_trans_length, _constructor_args, _solc_path=solc_path,
              _duplication=trans_duplication, use_rag=use_rag, api_key=api_key,
              llm_model=llm_model, llm_provider=llm_provider,
              ollama_endpoint=ollama_endpoint, ollama_thinking=ollama_thinking,
              use_llm_judge=use_llm_judge, judge_threshold=judge_threshold,
              generations=generations,
              openai_api_key=openai_api_key)
    if os.path.exists(res):
        shutil.copyfile(res, res_saved_path)  # move result json file to the specified path
        print(f"Kết quả fuzzing đã được lưu tại: {res_saved_path}")
    else:
        print(f"Cảnh báo: Không tìm thấy file kết quả tại {res}")


if __name__ == "__main__":
    PYTHON = sys.executable  # using current python interpreter
    FUZZER = "fuzzer/main.py"  # your fuzzer path in this repo
    cli()
    #test_run()
