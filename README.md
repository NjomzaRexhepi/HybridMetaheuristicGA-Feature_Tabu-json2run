# Hybrid Metaheuristic GA-Feature Tabu for Library Book Scanning Optimization

A hybrid metaheuristic approach combining Genetic Algorithms and Tabu Search for solving the library book scanning optimization problem.
This project implements a Genetic Algorithm to solve an optimization problem related to library book scanning (e.g., from the Hash Code challenge).

## Project Description

This project implements a hybrid metaheuristic solver that combines:
- Genetic Algorithm (GA) for global search
- Feature-based Tabu Search for local optimization
- GRASP (Greedy Randomized Adaptive Search Procedure) for initial solution generation

The solver is designed to optimize the order of library signups and book scanning to maximize the total score of scanned books within a given time frame.

## Overview
The GeneticAlgorithmSolver class:<br>
<ul>
  <li>Uses a population of candidate solutions.</li>
  <li>Selects parents with tournament selection.</li>
  <li>Applies two-point crossover to produce offspring.</li>
  <li>Applies mutation via a feature-based tabu search (local improvement).</li>
  <li>Uses elitism to carry over the best solution.</li>
  <li>Stops early if no improvement is observed over several generations.</li>
  <li>The solution encodes libraries to scan and books to select to maximize total score under given constraints.</li>
</ul>
## Key Features

- **Hybrid Approach**: Combines strengths of GA and Tabu Search
- **Feature-based Tabu Search**: Enhanced local search with move tracking
- **GRASP Initialization**: Generates high-quality starting solutions
- **Multiple Neighborhood Operators**: Various tweak methods for solution improvement
- **Adaptive Parameters**: Dynamic tournament selection and population management

## Requirements
Python: 3.8+<br>
Dependencies:
random
json
Modules: models, models.solver (containing InstanceData, Solution, and Solver classes)

Input Data:

Valid InstanceData object (parsed from input files)

Initial Solution object (can be generated via GRASP)

## Code Structure

The `GeneticAlgorithmSolver` class implements a genetic algorithm combined with local search (Tabu Search) to efficiently solve the problem. Its structure is as follows:

### `__init__` Method
Initializes the solver with:
- The problem instance.
- The initial solution.
- Genetic algorithm parameters such as population size, mutation probability, tournament size, and hill climbing steps.

### `load(file_path)` Method
Loads problem-specific data from a JSON file to prepare the instance for solving.

### `generate_initial_solution(instance)` Method
Generates an initial feasible solution using the GRASP (Greedy Randomized Adaptive Search Procedure) technique, providing a solid starting point for the genetic algorithm.

### `solve()` Method
The main loop of the genetic algorithm, which includes:
- **Selection**
- **Crossover**
- **Mutation using Tabu Search**
- **Elitism (keeping the best solution)**
- **Early stopping** based on the number of generations without improvement.

### `initialize_population(initial_solution)` Method
Creates the initial population by applying hill climbing to variations of the initial solution for diversity.

### `tournament_select(population)` Method
Performs tournament selection to choose parents for crossover, balancing exploration and exploitation.

### `crossover(parent1, parent2)` Method
Applies two-point crossover with validation to produce feasible and valid offspring solutions.

---

### Dependencies on `Solver` Class

The `GeneticAlgorithmSolver` relies on the `Solver` class to:
- Perform **Tabu Search** during mutation.
- Generate the **initial solution** using GRASP.

### Hyperparameter Optimization

We used [NEPS](https://github.com/automl/neps), a flexible and modular framework for neural architecture and hyperparameter search, to perform efficient hyperparameter optimization in our project. NEPS allowed us to easily define configuration spaces and utilize various search strategies.

## Installation

# Clone the repository:

git clone https://github.com/NjomzaRexhepi/HybridMetaheuristicGA-Feature_Tabu-json2run.git
cd HybridMetaheuristicGA-Feature_Tabu-json2run

##(Optional) Create a virtual environment:

conda create -n book_optimizer python=3.8
conda activate book_optimizer

## Academic Reference

This project is developed as part of the Algorithms Inspired by Nature 2025 course.

Course repository: https://github.com/ArianitHalimi/AIN_25

