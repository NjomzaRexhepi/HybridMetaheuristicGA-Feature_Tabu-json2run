import random
from typing import List, Tuple
 
import json
from models import InstanceData, Solution
from models.solver import Solver
import sys
 
class GeneticAlgorithmSolver:
        # def __init__(self, instance: InstanceData, initial_solution: Solution, 
        #          population_size=50, mutation_prob=1, 
        #          hill_climbing_steps=100, tabu_length=10):
        #     self.instance = instance
        #     self.initial_solution = initial_solution
        #     self.population_size = population_size
        #     self.tournament_size = max(2, population_size // 5)  # Dynamic tournament size
        #     self.mutation_prob = mutation_prob
        #     self.hill_climbing_steps = hill_climbing_steps
        #     self.tabu_length = tabu_length
        #     self.solver = Solver()

        def __init__(self, instance: InstanceData, initial_solution: Solution):
            self.instance = instance
            self.initial_solution = initial_solution
            self.population_size = 50
            self.tournament_size = 10
            self.mutation_prob = 1
            self.hill_climbing_steps = 100  # For mutation
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
        
        def tournament_select(self, population: List[Solution]) -> Solution:
            """Select best solution out of random tournament_size candidates"""
            tournament = random.sample(population, self.tournament_size)
            return max(tournament, key=lambda x: x.fitness_score)

        def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:

            def create_offspring(p1_signed, p2_signed):
                size = len(p1_signed)

                # Convert to sets for O(1) lookups
                p1_set = set(p1_signed)
                p2_set = set(p2_signed)

                # Create quick lookup for used libraries
                used_libs = set()

                # Initialize offspring with None
                offspring_signed = [None] * size

                # 1. Select random positions from parent1 (faster sampling)
                copy_indices = random.sample(range(size), k=size // 2)
                for idx in copy_indices:
                    lib = p1_signed[idx]
                    offspring_signed[idx] = lib
                    used_libs.add(lib)

                # 2. Prepare parent2 libraries not yet used
                available_p2 = [lib for lib in p2_signed if lib not in used_libs]
                p2_ptr = 0

                # 3. Fill remaining positions
                remaining_indices = [i for i, lib in enumerate(offspring_signed) if lib is None]

                for idx in remaining_indices:
                    if p2_ptr < len(available_p2):
                        offspring_signed[idx] = available_p2[p2_ptr]
                        p2_ptr += 1
                    else:
                        # Fallback to unused libraries from parent1
                        remaining_p1 = [lib for lib in p1_signed if lib not in used_libs]
                        if remaining_p1:
                            offspring_signed[idx] = random.choice(remaining_p1)
                            used_libs.add(offspring_signed[idx])
                        else:
                            # If all libraries are used (shouldn't happen with valid parents)
                            unused = list(p1_set - used_libs)
                            if unused:
                                offspring_signed[idx] = random.choice(unused)
                            else:
                                raise ValueError("Crossover failed - no available libraries")

                return offspring_signed

            # Create base signed orders
            try:
                offspring1_signed = create_offspring(parent1.signed_libraries, parent2.signed_libraries)
                offspring2_signed = create_offspring(parent2.signed_libraries, parent1.signed_libraries)

                # Create complete solutions
                def build_solution(signed_libs):
                    scanned_books = set()
                    scanned_per_lib = {}
                    used_libs = []

                    current_day = 0
                    for lib in signed_libs:
                        lib_data = self.instance.libs[lib]
                        if current_day + lib_data.signup_days > self.instance.num_days:
                            continue

                        current_day += lib_data.signup_days
                        remaining_days = self.instance.num_days - current_day
                        max_books = remaining_days * lib_data.books_per_day

                        available_books = [b.id for b in lib_data.books
                                        if b.id not in scanned_books]
                        available_books.sort(key=lambda x: self.instance.scores[x], reverse=True)
                        selected = available_books[:max_books]

                        if selected:
                            scanned_books.update(selected)
                            scanned_per_lib[lib] = selected
                            used_libs.append(lib)

                    return Solution(
                        signed_libs=used_libs,
                        unsigned_libs=list(set(range(self.instance.num_libs)) - set(used_libs)),
                        scanned_books_per_library=scanned_per_lib,
                        scanned_books=scanned_books
                    )

                return (build_solution(offspring1_signed),
                        build_solution(offspring2_signed))

            except ValueError as e:
                # Fallback to parents if crossover fails
                print(f"Crossover failed: {e}, returning parents")
                return parent1, parent2

