import random
from typing import List, Tuple
 
import json
from models import InstanceData, Solution
from models.solver import Solver
import sys

 
 
class GeneticAlgorithmSolver:
        def __init__(self, instance: InstanceData, initial_solution: Solution, 
                 population_size=50, mutation_prob=1, 
                 hill_climbing_steps=100, tabu_length=10):
            self.instance = instance
            self.initial_solution = initial_solution
            self.population_size = population_size
            self.tournament_size = max(2, population_size // 5)  # Dynamic tournament size
            self.mutation_prob = mutation_prob
            self.hill_climbing_steps = hill_climbing_steps
            self.tabu_length = tabu_length
            self.solver = Solver()
 
        def load(self, file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
 
        def generate_initial_solution(self, instance):
            """Generate initial solution using GRASP"""
            return self.solver.generate_initial_solution_grasp(instance, max_time=30)
    
        def solve(self) -> Solution:
            # Initialize population with slight variations of initial solution
            population = self.initialize_population(self.initial_solution)
    
            best_solution = max(population, key=lambda x: x.fitness_score)  # Track the best solution
    
            for generation in range(self.population_size):  # Max generations
                # Evaluate population: use the best_solution instead of sorting the entire population
                print(f"Gen {generation}: Best fitness = {best_solution.fitness_score}")
    
                # Create new generation
                new_population = [best_solution]  # Keep the best solution
    
                while len(new_population) < self.population_size:
                    # Selection
                    parent1 = self.tournament_select(population)
                    parent2 = self.tournament_select(population)
    
                    # Crossover
                    offspring1, offspring2 = self.crossover(parent1, parent2)
    
                    # Apply mutation with a smaller probability to avoid costly operations
                    if random.random() < self.mutation_prob:
                        offspring1 = self.solver.feature_based_tabu_search(
                            offspring1, self.instance, max_iterations=self.hill_climbing_steps
                        )
    
                    if random.random() < self.mutation_prob:
                        offspring2 = self.solver.feature_based_tabu_search(
                            offspring2, self.instance, max_iterations=self.hill_climbing_steps
                        )
    
                    # Add offspring to the new population
                    new_population.extend([offspring1, offspring2])
    
                # Limit population size (trim excess individuals)
                population = new_population[:self.population_size]
    
                # Update the best solution
                best_solution = max(population, key=lambda x: x.fitness_score)
    
                # Early stopping condition: check if no significant improvement is made in X generations
                if generation > 10 and best_solution.fitness_score == max(population, key=lambda x: x.fitness_score).fitness_score:
                    print(f"Early stopping: No significant improvement in generation {generation}")
                    break
    
            return best_solution
    
    
        def initialize_population(self, initial_solution: Solution) -> List[Solution]:
            """Create initial population with variations of the initial solution"""
            population = [self.initial_solution]
    
            while len(population) < self.population_size:
                # Create variant by shuffling some libraries
                variant_fitness, variant = self.solver.hill_climbing_combined_w_initial_solution(initial_solution, self.instance, iterations=5)
                population.append(variant)
    
            return population

