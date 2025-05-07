from models import genetic_solver, solution, instance_data, library
import argparse
import json
import time
import os

def load_instance_data(file_path):
    with open(file_path, 'r') as file:
        # Read the first line: num_books, num_libs, num_days
        num_books, num_libs, num_days = map(int, file.readline().split())
        
        # Read the second line: book scores
        scores = list(map(int, file.readline().split()))

        # Initialize list of libraries
        libs = []
        
        # Read library details
        for lib_id in range(num_libs):
            # Read library details: num_books, signup_days, books_per_day
            num_books_in_lib, signup_days, books_per_day = map(int, file.readline().split())
            
            # Read the books for this library
            books = list(map(int, file.readline().split()))
            
            # Instantiate the library using the constructor that includes book_scores
            libs.append(library.Library(num_books_in_lib, signup_days, books_per_day, books, scores))

    return instance_data.InstanceData(num_books, num_libs, num_days, scores, libs)

def create_initial_solution(instance):
    # Create a simple initial solution (can be improved)
    solver = genetic_solver.GeneticAlgorithmSolver(instance, None)
    return solver.generate_initial_solution(instance)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--population_size", type=int, required=True)
    parser.add_argument("--mutation_prob", type=float, required=True)
    parser.add_argument("--hill_climbing_steps", type=int, required=True)
    parser.add_argument("--tabu_length", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=False)

    args = parser.parse_args()

    # Load instance data
    input_path = os.path.join('input', args.input_file)
    instance = load_instance_data(input_path)

    # Create initial solution
    initial_solution = create_initial_solution(instance)

    # Instantiate the solver with parameters
    solver = genetic_solver.GeneticAlgorithmSolver(
        instance=instance,
        initial_solution=initial_solution,
        population_size=args.population_size,
        mutation_prob=args.mutation_prob,
        hill_climbing_steps=args.hill_climbing_steps,
        tabu_length=args.tabu_length
    )

    # Solve the problem and measure time
    start_time = time.time()
    solution = solver.solve()
    runtime = time.time() - start_time

    # Prepare results
    results = {
        "input_file": args.input_file,
        "fitness_score": solution.fitness_score,
        "runtime": runtime,
        "parameters": {
            "population_size": args.population_size,
            "mutation_prob": args.mutation_prob,
            "hill_climbing_steps": args.hill_climbing_steps,
            "tabu_length": args.tabu_length
        }
    }

    # Output results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f)
    else:
        print(json.dumps(results, indent=2))