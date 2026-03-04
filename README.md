# MAESFuzz
A Multi-Agent and Explainable Semantic-Guided Smart Contract Fuzzing with LLMs

## Installation

**Requirements**
- macOS or Ubuntu 20.04+
- Python 3.10 – 3.13

```shell
git clone https://github.com/VNUHCM-UIT-InSecLab/MAESFuzz.git
cd MAESFuzz/

python3 -m venv venv
source venv/bin/activate

pip install wheel
pip install -r requirements.txt

# Install Solidity compiler(s) you need
solc-select install 0.4.26
solc-select install 0.8.26

mkdir -p result/ log/
```

**Configure solc path** — open `config.py` and set `SOLC_BIN_PATH` to the binary matching your default version:

```python
# config.py
SOLC_BIN_PATH = "/Users/<you>/.solc-select/artifacts/solc-0.4.26/solc-0.4.26"
```

> Tip: run `solc-select artifacts` to list installed paths, or `which solc` if using a system install.

---

## Quick Start

```shell
# Always use the venv interpreter directly to ensure all dependencies are available
venv/bin/python3.13 MAESFuzz.py examples/reentrance.sol Reentrance \
  --solc-version 0.4.26 \
  --fuzz-time 60 \
  --openai-api-key YOUR_OPENAI_API_KEY \
  --provider openai \
  --model gpt-4o
```

After fuzzing you will see:

```
[1/4] Analyzer …      ← static analysis + LLM dataflow
[2/4] Generator …     ← RAG-enhanced seed synthesis
[3/4] Executor …      ← evolutionary fuzzing loop
[4/4] Reporter …      ← LLM-as-a-Judge audit report

============================================================
  Report saved: reports/Reentrance/fuzz_report_<timestamp>.md
============================================================
```

---

## Usage

### Basic syntax

```shell
venv/bin/python3.13 MAESFuzz.py <contract.sol> <ContractName> [options]
```

### Common options

| Option | Description | Default |
|--------|-------------|---------|
| `--solc-version <ver>` | Solidity compiler version | `0.8.26` |
| `--fuzz-time <sec>` | Fuzzing time budget | `60` |
| `-g, --generations <n>` | Fixed generation count (overrides time) | — |
| `--max-trans-length <n>` | Max transaction sequence length | `10` |
| `--no-rag` | Disable RAG-enhanced seed generation | RAG on |
| `--openai-api-key <key>` | OpenAI API key | env `OPENAI_API_KEY` |
| `--api-key <key>` | Google / Gemini API key | env `GOOGLE_API_KEY` |
| `--model <name>` | LLM model name | `gpt-4o` |
| `--provider <name>` | `openai` \| `gemini` \| `ollama` | `openai` |
| `--ollama-endpoint <url>` | Ollama server URL | `http://127.0.0.1:11434` |
| `--result-path <path>` | Output JSON path | `result/results.json` |
| `--constructor-params <path>` | Constructor params JSON, or `auto` | `auto` |

### Examples

**Fuzz with OpenAI GPT-4o:**
```shell
venv/bin/python3.13 MAESFuzz.py examples/reentrance.sol Reentrance \
  --solc-version 0.4.26 \
  --openai-api-key sk-proj-... \
  --provider openai --model gpt-4o
```

**Fuzz with Gemini:**
```shell
venv/bin/python3.13 MAESFuzz.py examples/reentrance.sol Reentrance \
  --solc-version 0.4.26 \
  --api-key AIza... \
  --provider gemini --model gemini-2.0-flash
```

**Fuzz with local Ollama (no API key needed):**
```shell
venv/bin/python3.13 MAESFuzz.py examples/reentrance.sol Reentrance \
  --solc-version 0.4.26 \
  --provider ollama --model deepseek-r1:7b
```

**Fixed generation count instead of time:**
```shell
venv/bin/python3.13 MAESFuzz.py examples/reentrance.sol Reentrance \
  --solc-version 0.4.26 -g 30 \
  --openai-api-key sk-proj-...
```

**With custom constructor parameters:**
```shell
# constructor_params.json format:
# { "_param1": { "type": "uint256", "value": 12 } }

venv/bin/python3.13 MAESFuzz.py Token.sol Token \
  --solc-version 0.8.0 \
  --constructor-params constructor_params.json \
  --openai-api-key sk-proj-...
```

---

## API Key Configuration

Instead of passing keys on every command, create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=AIza...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
```

Then export before running (or add to your shell profile):

```shell
export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d= -f2)
venv/bin/python3.13 MAESFuzz.py examples/reentrance.sol Reentrance --solc-version 0.4.26
```

---

## Output

| Path | Content |
|------|---------|
| `result/results.json` | Coverage metrics, detected bugs (raw JSON) |
| `reports/<Contract>/fuzz_report_<ts>.md` | LLM-as-a-Judge audit report (Markdown) |
| `log/INFO.log` | Execution log |

Example `results.json`:
```json
{
  "Reentrance": {
    "errors": { "107": ["Reentrancy"] },
    "code_coverage":   { "percentage": 99.73, "covered": 372, "total": 373 },
    "branch_coverage": { "percentage": 100.0,  "covered": 18,  "total": 18 }
  }
}
```

---

## Repository Policy
- Do **not** commit `venv/` or `.env` files with API keys

## Citation
If you use this repository in your research, please cite the corresponding MAESFuzz paper.