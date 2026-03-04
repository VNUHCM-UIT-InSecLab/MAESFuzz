# MAESFuzz
A Multi-Agent and Explainable Semantic-Guided Smart Contract Fuzzing with LLMs
### Installation
* OS: macOS / Ubuntu 20.04 LTS
* Python (>= 3.8, < 3.13)
* Install env and dependencies, follow the instructions below:

```shell
git clone https://github.com/VNUHCM-UIT-InSecLab/MAESFuzz.git
cd MAESFuzz/
python3 -m venv venv
source venv/bin/activate
pip install wheel
pip install -r requirements.txt
solc-select install 0.4.26
solc-select use 0.4.26
mkdir -p result/
```

* config your solc bin path in `config.py` in `SOLC_BIN_PATH` variable.
    * tips: you can use `which solc` to get the path of your solc bin. For `solc-select`, it is usually located at `~/.solc-select/artifacts/solc-0.4.26/solc-0.4.26` or simply inside `venv/bin/solc`.

### Usage

* Config `MAESFuzz.py`, you can run a simple demo when you first run it. Follow the code below and run `test_run()`
  function instead of `cli()`. This demo will run a simple fuzzer on `examples/reentrance.sol` and generate test cases.

```python
if __name__ == "__main__":
    PYTHON = sys.executable  # your python interpreter path
    FUZZER = "fuzzer/main.py"  # your fuzzer path in this repo
    # cli()
    test_run()
```

* Then, you can see the following output:

```shell
INFO:Fuzzer  :LLM evolution enabled: model=gpt-4o, provider=openai
INFO:Detector:-----------------------------------------------------
INFO:Detector:          !!! Reentrancy detected !!!         
INFO:Detector:-----------------------------------------------------
INFO:Detector:SWC-ID:   107
INFO:Detector:Severity: High
INFO:Detector:-----------------------------------------------------
INFO:Detector:Source code line:
INFO:Detector:-----------------------------------------------------
INFO:Detector:examples/reentrance.sol:24:1
INFO:Detector:msg.sender.call.value(_amount)()
...
INFO:Analysis:-----------------------------------------------------
INFO:Analysis:Number of generations:     11
INFO:Analysis:Number of transactions:    441 (118 unique)
INFO:Analysis:Transactions per second:   1153
INFO:Analysis:Total code coverage:       90.00%
INFO:Analysis:Total branch coverage:     87.50%
INFO:Analysis:Total execution time:      0.38 seconds
INFO:Analysis:Total memory consumption:  555.06 MB
Kết quả fuzzing đã được lưu tại: result/results.json
```

* You can see the result in the corresponding `results.json` JSON file (default `result/results.json` or `result/res.json`). Part of the result is as follows:

```json
{
  "Reentrance": {
    "errors": {
      "107": [
        "Reentrancy"
      ]
    },
    ...
    "code_coverage": {
      "percentage": 90.0,
      "covered": 45,
      "total": 50
    }
  }
}
```

#### Test your contract by command line

Since MAESFuzz is primarily an LLM-guided fuzzing tool, passing your API keys to the system is the recommended way to achieve the highest performance. By default, MAESFuzz supports OpenAI's GPT models and Google's Gemini models natively.

* Config `MAESFuzz.py` to command line mode by ensuring `cli()` is uncommented:

```python
if __name__ == "__main__":
    PYTHON = sys.executable 
    FUZZER = "fuzzer/main.py" 
    cli()
```

* Suppose the smart contract file you want to test is `examples/reentrance.sol`, the name of the contract under tested is `Reentrance`, the version of solc compiler is `0.4.26`, the max length of transaction sequence is `5`, and fuzz time is `60` seconds.

  Then you can run the following command to test it (ensure your virtual environment is active):

```shell
./venv/bin/python MAESFuzz.py examples/reentrance.sol Reentrance --solc-version 0.4.26 --max-trans-length 5 --fuzz-time 60 --provider openai --model gpt-4o --openai-api-key YOUR_OPENAI_API_KEY
```

* You can manually set the constructor arguments of the contract under tested by changing the `--constructor-params` parameter. Specifically, you can pass the path to a json file which contains the constructor parameters of the contract under test. The format of the json file is as follows:

```json
{
  "_param1": {
    "type": "uint256",
    "value": 12
  }
}
```

Then you can run the following command to test it with the constructor arguments alongside the LLM flags:

```shell
./venv/bin/python MAESFuzz.py examples/reentrance.sol Reentrance --constructor-params custom_params.json --provider openai --model gpt-4o --openai-api-key YOUR_OPENAI_API_KEY
```

* You can configure API keys conveniently via a local `.env` file instead of repeating them in CLI constraints:
```env
OPENAI_API_KEY=sk-proj-xyz...
GOOGLE_API_KEY=AIza...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
```

* After the fuzzing process is finished, you can see the result in `result/results.json` or your specific generated output file.

### Repository Policy
- Do NOT commit `node_modules/` or `venv/`
- Do NOT commit `.env` files with API keys

### Links and Citation
* If you use this repository in your research, please provide appropriate citation based on the published MAESFuzz/UniFuzz paper.