# Hybrid Metaheuristic GA-Feature Tabu for Library Book Scanning Optimization

A hybrid metaheuristic approach combining Genetic Algorithms and Tabu Search for solving the library book scanning optimization problem.

## Project Description

This project implements a hybrid metaheuristic solver that combines:
- Genetic Algorithm (GA) for global search
- Feature-based Tabu Search for local optimization
- GRASP (Greedy Randomized Adaptive Search Procedure) for initial solution generation

The solver is designed to optimize the order of library signups and book scanning to maximize the total score of scanned books within a given time frame.

## Key Features

- **Hybrid Approach**: Combines strengths of GA and Tabu Search
- **Feature-based Tabu Search**: Enhanced local search with move tracking
- **GRASP Initialization**: Generates high-quality starting solutions
- **Multiple Neighborhood Operators**: Various tweak methods for solution improvement
- **Adaptive Parameters**: Dynamic tournament selection and population management

## Requirements
Python: 3.8+

Input Data:

Valid InstanceData object (parsed from input files)

Initial Solution object (can be generated via GRASP)

### Hyperparameter Optimization

We used [NEPS](https://github.com/automl/neps), a flexible and modular framework for neural architecture and hyperparameter search, to perform efficient hyperparameter optimization in our project. NEPS allowed us to easily define configuration spaces and utilize various search strategies.

## Installation

# Clone the repository:

git clone https://github.com/NjomzaRexhepi/HybridMetaheuristicGA-Feature_Tabu-json2run.git
cd HybridMetaheuristicGA-Feature_Tabu-json2run

##(Optional) Create a virtual environment:

conda create -n book_optimizer python=3.8
conda activate book_optimizer

# Install required libraries:

pip install -r requirements.txt

## Academic Reference

This project is developed as part of the Algorithms Inspired by Nature 2025 course.

Course repository: https://github.com/ArianitHalimi/AIN_25

