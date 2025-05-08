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

## Installation
1. Clone the repository:
   https://github.com/NjomzaRexhepi/HybridMetaheuristicGA-Feature_Tabu-json2run.git 

Algorithms inspired by nature 2025 class repository available on https://github.com/ArianitHalimi/AIN_25

