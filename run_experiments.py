import subprocess
import json
import os
from pathlib import Path

def run_experiments(config_file):
    # Convert to absolute path
    config_file = os.path.abspath(config_file)
    config_dir = os.path.dirname(config_file)
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Use paths relative to the config file location
    input_dir = os.path.join(config_dir, "input")
    output_dir = os.path.join(config_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    for input_file in config["input_files"]:
        input_path = os.path.join(input_dir, input_file)
        
        for params in config["parameter_combinations"]:
            for repeat in range(config["common_params"]["num_repeats"]):
                output_file = os.path.join(
                    output_dir,
                    f"{Path(input_file).stem}_ps{params['population_size']}_"
                    f"mp{params['mutation_prob']}_hc{params['hill_climbing_steps']}_"
                    f"tl{params['tabu_length']}_run{repeat}.json"
                )
                
                cmd = [
                    "python", 
                    os.path.join(config_dir, "main.py"),
                    "--input_file", input_path,
                    "--population_size", str(params["population_size"]),
                    "--mutation_prob", str(params["mutation_prob"]),
                    "--hill_climbing_steps", str(params["hill_climbing_steps"]),
                    "--tabu_length", str(params["tabu_length"]),
                    "--output_file", output_file
                ]
                
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # Use absolute path to the config file
    config_path = os.path.join(os.path.dirname(__file__), "experiments.j2son")
    run_experiments(config_path)