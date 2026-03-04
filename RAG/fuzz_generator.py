import json

class CrossFuzzConfigGenerator:
    @staticmethod
    def generate_config(analysis):
        config = {
            "target_contract": "MainContract",
            "solc_version": "0.8.19",
            "fuzzing_params": {
                "max_seq_len": 10,
                "focus_functions": [],
                "critical_vars": [],
                "custom_seeds": []
            }
        }
        
        for vuln in analysis.get("detected_vulnerabilities", []):
            config["fuzzing_params"]["focus_functions"].extend(
                vuln["affected_functions"]
            )
            config["fuzzing_params"]["critical_vars"].extend(
                vuln["critical_variables"]
            )
            config["fuzzing_params"]["custom_seeds"].append({
                "vulnerability": vuln["type"],
                "inputs": vuln["fuzzing_inputs"]
            })
            
        return config

    @staticmethod
    def save_config(config, output_path):
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
