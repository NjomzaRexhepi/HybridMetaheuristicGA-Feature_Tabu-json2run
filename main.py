# # from models import genetic_solver, solution, instance_data, library
# # import argparse
# # import json
# # import time
# # import os

# # def load_instance_data(file_path):
# #     with open(file_path, 'r') as file:
# #         # Read the first line: num_books, num_libs, num_days
# #         num_books, num_libs, num_days = map(int, file.readline().split())
        
# #         # Read the second line: book scores
# #         scores = list(map(int, file.readline().split()))

# #         # Initialize list of libraries
# #         libs = []
        
# #         # Read library details
# #         for lib_id in range(num_libs):
# #             # Read library details: num_books, signup_days, books_per_day
# #             num_books_in_lib, signup_days, books_per_day = map(int, file.readline().split())
            
# #             # Read the books for this library
# #             books = list(map(int, file.readline().split()))
            
# #             # Instantiate the library using the constructor that includes book_scores
# #             libs.append(library.Library(num_books_in_lib, signup_days, books_per_day, books, scores))

# #     return instance_data.InstanceData(num_books, num_libs, num_days, scores, libs)

# # def create_initial_solution(instance):
# #     # Create a simple initial solution (can be improved)
# #     solver = genetic_solver.GeneticAlgorithmSolver(instance, None)
# #     return solver.generate_initial_solution(instance)

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--input_file", type=str, required=True)
# #     parser.add_argument("--population_size", type=int, required=True)
# #     parser.add_argument("--mutation_prob", type=float, required=True)
# #     parser.add_argument("--hill_climbing_steps", type=int, required=True)
# #     parser.add_argument("--tabu_length", type=int, required=True)
# #     parser.add_argument("--output_file", type=str, required=False)

# #     args = parser.parse_args()

# #     # Load instance data
# #     input_path = os.path.join('input', args.input_file)
# #     instance = load_instance_data(input_path)

# #     # Create initial solution
# #     initial_solution = create_initial_solution(instance)

# #     # Instantiate the solver with parameters
# #     solver = genetic_solver.GeneticAlgorithmSolver(
# #         instance=instance,
# #         initial_solution=initial_solution,
# #         population_size=args.population_size,
# #         mutation_prob=args.mutation_prob,
# #         hill_climbing_steps=args.hill_climbing_steps,
# #         tabu_length=args.tabu_length
# #     )

# #     # Solve the problem and measure time
# #     start_time = time.time()
# #     solution = solver.solve()
# #     runtime = time.time() - start_time

# #     # Prepare results
# #     results = {
# #         "input_file": args.input_file,
# #         "fitness_score": solution.fitness_score,
# #         "runtime": runtime,
# #         "parameters": {
# #             "population_size": args.population_size,
# #             "mutation_prob": args.mutation_prob,
# #             "hill_climbing_steps": args.hill_climbing_steps,
# #             "tabu_length": args.tabu_length
# #         }
# #     }

# #     # Output results
# #     if args.output_file:
# #         with open(args.output_file, 'w') as f:
# #             json.dump(results, f)
# #     else:
# #         print(json.dumps(results, indent=2))



###uncomment if u want ot use neps
from datetime import datetime
from functools import partial
 
from models import genetic_solver, solution, instance_data, library
import argparse
import json
import time
import os
import neps
import torch
import logging
 
 
def load_instance_data(file_path):
    with open(file_path, 'r') as file:
        num_books, num_libs, num_days = map(int, file.readline().split())
 
        scores = list(map(int, file.readline().split()))
 
        libs = []
 
        for lib_id in range(num_libs):
            num_books_in_lib, signup_days, books_per_day = map(int, file.readline().split())
 
            books = list(map(int, file.readline().split()))
 
            libs.append(library.Library(num_books_in_lib, signup_days, books_per_day, books, scores))
 
    return instance_data.InstanceData(num_books, num_libs, num_days, scores, libs)
 
 
def create_initial_solution(instance):
    solver = genetic_solver.GeneticAlgorithmSolver(instance, None)
    return solver.generate_initial_solution(instance)
 
 
def run_pipeline_for_neps(
        population_size,
        mutation_prob,
        hill_climbing_steps,
        tabu_length,
        problem_instance,
        initial_sol,
        input_file_name
):
    print(f"\n[NePS EVAL] Starting evaluation with params:")
    print(f"  population_size: {population_size}")
    print(f"  mutation_prob: {mutation_prob:.4f}")
    print(f"  hill_climbing_steps: {hill_climbing_steps}")
    print(f"  tabu_length: {tabu_length}")
 
    solver = genetic_solver.GeneticAlgorithmSolver(
        instance=problem_instance,
        initial_solution=initial_sol,
        population_size=population_size,
        mutation_prob=mutation_prob,
        hill_climbing_steps=hill_climbing_steps,
        tabu_length=tabu_length
    )
 
    start_time = time.time()
    solved_solution = solver.solve()
    runtime = time.time() - start_time
 
    fitness_score = solved_solution.fitness_score
    print(f"[NePS EVAL] Fitness: {fitness_score}, Runtime: {runtime:.2f}s")
 
    # NePS only minimizes, so we return -fitness_score is we want to maximaze 
    return {
        "loss": -fitness_score,  # qetu kom ndryshi prej fitness_score ne negativ
        # changed from  -fitness_score -> fitness_score
        "cost": runtime,
        # Store extra information
        "info_dict": {
            "fitness_score": fitness_score,
            "input_file": input_file_name,
            "population_size": population_size,
            "mutation_prob": mutation_prob,
            "hill_climbing_steps": hill_climbing_steps,
            "tabu_length": tabu_length,
            "runtime": runtime
        }
    }
 
 
if __name__ == "__main__":
    print("Cuda:", torch.cuda.is_available())
    print("Running Neural Pipeline Search for Genetic Algorithm Hyperparameter Optimization")
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm with NePS hyperparameter optimization.")
    parser.add_argument("--input_file", type=str, default="B2.5k_L25_D50.txt", help="Input File")
    parser.add_argument("--max_evaluations", type=int, default=10,
                        help="Maximum number of hyperparameter configurations to try.")
    parser.add_argument("--neps_root_dir", type=str, default="neps_results_B2.5k_L25_D50_output",
                        help="Directory to store NePS results.")
 
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
 
    input_path = os.path.join('input', args.input_file)
 



    log_directory = 'results'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    current_date = datetime.now().strftime("%d-%m-%Y")
    log_file_name = f'{log_directory}/{args.input_file}_{current_date}.log'
    logging.basicConfig(filename=log_file_name,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    instance = load_instance_data(input_path)
 
    initial_solution = create_initial_solution(instance)
 
    # better for smaller files
    # pipeline_space = dict(
    #     population_size=neps.IntegerParameter(lower=10, upper=500, log=False),  
    #     mutation_prob=neps.FloatParameter(lower=0.01, upper=0.8, log=False),  
    #     hill_climbing_steps=neps.IntegerParameter(lower=10, upper=100, log=False),
    #     # Fewer steps to avoid long refinements
    #     tabu_length=neps.IntegerParameter(lower=0, upper=20, log=False)  
    # )
 
    # better for medium or larger files
    pipeline_space = dict(
        population_size=neps.IntegerParameter(lower=10, upper=100, log=False),
        mutation_prob=neps.FloatParameter(lower=0.01, upper=0.5, log=False),
        hill_climbing_steps=neps.IntegerParameter(lower=0, upper=40, log=False),
        tabu_length=neps.IntegerParameter(lower=0, upper=15, log=False)
    )


    run_pipeline_with_fixed_args = partial(
        run_pipeline_for_neps,
        problem_instance=instance,
        initial_sol=initial_solution,
        input_file_name=args.input_file
    )
 
    print(f"\nStarting NePS hyperparameter search...")
    logging.info(f"\nStarting NePS hyperparameter search...")
 
    print(f"Max evaluations: {args.max_evaluations}")
    logging.info(f"Max evaluations: {args.max_evaluations}")
 
    print(f"Results will be stored in: {args.neps_root_dir}")
    logging.info(f"Results will be stored in: {args.neps_root_dir}")

 
    neps.run(
        run_pipeline=run_pipeline_with_fixed_args,
        pipeline_space=pipeline_space,
        root_directory=args.neps_root_dir,
        max_evaluations_total=args.max_evaluations,
    )
 
    print("\nNePS search finished.")
    logging.info("\nNePS search finished.")
 
    print(f"Check the directory '{args.neps_root_dir}' for detailed results and logs.")
    logging.info(f"Check the directory '{args.neps_root_dir}' for detailed results and logs.")
 
    # Try to read the results from NePS output directory
    try:
        all_configs_and_losses = neps.status(args.neps_root_dir)
 
        best_loss = float('inf')
        best_config_details = None
 
        for config_id, config_data in all_configs_and_losses[0].items():
            if config_data.result is None:
                continue
            if 'loss' in config_data.result and config_data.result['loss'] < best_loss:
                best_loss = config_data.result['loss']
                best_config_details = {
                    "config_id": config_id,
                    "loss": config_data.result['loss'],
                    "config": config_data.config,
                    "info": config_data.result.get('info_dict', {})
                }
 
        if best_config_details:
            print("\n--- Best Configuration Found by NePS ---")
            logging.info("\n--- Best Configuration Found by NePS ---")
            print(f"Config ID: {best_config_details['config_id']}")
            logging.info(f"Config ID: {best_config_details['config_id']}")
            print(f"Best Loss: {best_config_details['loss']:.4f}")
            logging.info(f"Best Loss (Minimized -Fitness): {best_config_details['loss']:.4f}")
            if 'fitness_score' in best_config_details['info']:
                print(f"Corresponding Fitness Score: {best_config_details['info']['fitness_score']:.4f}")
                logging.info(f"Corresponding Fitness Score: {best_config_details['info']['fitness_score']:.4f}")
            print("Hyperparameters:")
            logging.info("Hyperparameters:")
            for key, value in best_config_details['config'].items():
                print(f"  {key}: {value}")
                logging.info(f"  {key}: {value}")
            if 'runtime' in best_config_details['info']:
                print(f"Runtime for this config: {best_config_details['info']['runtime']:.2f}s")
                logging.info(f"Runtime for this config: {best_config_details['info']['runtime']:.2f}s")
        else:
            print("No completed configurations found by NePS.")
            logging.error("No completed configurations found by NePS.")
 
    except FileNotFoundError:
        print(f"Could not load NePS status. Directory '{args.neps_root_dir}' might be empty or structured differently.")
        logging.error(f"Could not load NePS status. Directory '{args.neps_root_dir}' might be empty or structured differently.")
    except Exception as e:
        print(f"Error processing NePS results: {e}")
        logging.error(f"Error processing NePS results: {e}")





