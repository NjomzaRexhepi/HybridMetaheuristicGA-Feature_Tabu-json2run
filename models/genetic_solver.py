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
            population = self.initialize_population(self.initial_solution)
            best_solution = min(population, key=lambda x: x.fitness_score)
            best_score = best_solution.fitness_score

            no_improvement_counter = 0
            patience = 10  # adjustable

            for generation in range(self.population_size):
                print(f"Gen {generation}: Best fitness = {best_score}")

                new_population = [best_solution]  # Elitism

                while len(new_population) < self.population_size:
                    parent1 = self.tournament_select(population)
                    parent2 = self.tournament_select(population)

                    offspring1, offspring2 = self.crossover(parent1, parent2)

                    for offspring in (offspring1, offspring2):
                        if random.random() < self.mutation_prob:
                            offspring = self.solver.feature_based_tabu_search(
                                offspring, self.instance, max_iterations=self.hill_climbing_steps
                            )
                        new_population.append(offspring)

                population = new_population[:self.population_size]
                current_best = min(population, key=lambda x: x.fitness_score)

                if current_best.fitness_score < best_score:
                    best_solution = current_best
                    best_score = current_best.fitness_score
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

                if no_improvement_counter >= patience:
                    print(f"Early stopping: No improvement for {patience} generations.")
                    break

            return best_solution

        # def solve(self) -> Solution:
        #     # Initialize population
        #     population = self.initialize_population(self.initial_solution)
        #     best_solution = min(population, key=lambda x: x.fitness_score)

        #     # Trackers
        #     best_fitness = best_solution.fitness_score
        #     stagnation_counter = 0
        #     max_stagnation = 10
        #     elite_count = 2

        #     for generation in range(self.population_size * 3):  # Allow more generations
        #         print(f"Gen {generation}: Best fitness = {best_solution.fitness_score}")

        #         # Sort population by fitness (descending)
        #         population.sort(key=lambda x: x.fitness_score, reverse=True)

        #         # Keep elites
        #         new_population = population[:elite_count]

        #         # Adaptive mutation probability (more exploration early on)
        #         mutation_prob = min(0.05, 1.0 - generation / (self.population_size * 2))

        #         # Generate rest of population
        #         while len(new_population) < self.population_size:
        #             # Hybrid parent selection
        #             if random.random() < 0.8:
        #                 parent1 = self.tournament_select(population)
        #                 parent2 = self.tournament_select(population)
        #             else:
        #                 parent1 = random.choice(population)
        #                 parent2 = random.choice(population)

        #             # Crossover
        #             offspring1, offspring2 = self.crossover(parent1, parent2)

        #             # Mutate offspring with adaptive probability
        #             if random.random() < mutation_prob:
        #                 offspring1 = self.solver.feature_based_tabu_search(
        #                     offspring1, self.instance, max_iterations=self.hill_climbing_steps
        #                 )

        #             if random.random() < mutation_prob:
        #                 offspring2 = self.solver.feature_based_tabu_search(
        #                     offspring2, self.instance, max_iterations=self.hill_climbing_steps
        #                 )

        #             new_population.extend([offspring1, offspring2])

        #         # Trim to population size
        #         population = new_population[:self.population_size]

        #         # Track best solution
        #         current_best = min(population, key=lambda x: x.fitness_score)
        #         if current_best.fitness_score > best_fitness:
        #             best_solution = current_best
        #             best_fitness = current_best.fitness_score
        #             stagnation_counter = 0
        #         else:
        #             stagnation_counter += 1

        #         # Early stopping if no improvement
        #         if stagnation_counter >= max_stagnation:
        #             print(f"Early stopping at generation {generation} due to stagnation.")
        #             break

        #         # Optional: inject diversity if totally stuck
        #         if generation > 0 and generation % 20 == 0:
        #             print("Injecting diversity...")
        #             population[-5:] = self.initialize_population(self.initial_solution)[:5]

        #     return best_solution

    
    
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
            return min(tournament, key=lambda x: x.fitness_score)

        def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
            def two_point_crossover(p1_signed, p2_signed):
                size = len(p1_signed)
                
                if size < 2:
                    # If crossover is not possible, just return a copy
                    return p1_signed.copy()

                point1 = random.randint(0, size - 2)
                point2 = random.randint(point1 + 1, size - 1)

                offspring = [None] * size
                offspring[point1:point2] = p1_signed[point1:point2]
                used = set(offspring[point1:point2])

                p2_idx = 0
                for i in range(size):
                    if offspring[i] is None:
                        while p2_idx < size and p2_signed[p2_idx] in used:
                            p2_idx += 1
                        if p2_idx < size:
                            offspring[i] = p2_signed[p2_idx]
                            used.add(p2_signed[p2_idx])
                            p2_idx += 1
                        else:
                            # If no valid gene is left to copy (rare), fill with a placeholder
                            offspring[i] = -1  # Or another appropriate fallback

                return offspring

            try:
                if len(parent1.signed_libraries) < 2 or len(parent2.signed_libraries) < 2:
                    print("Crossover skipped due to small parent size.")
                    return parent1, parent2

                offspring1_signed = two_point_crossover(parent1.signed_libraries, parent2.signed_libraries)
                offspring2_signed = two_point_crossover(parent2.signed_libraries, parent1.signed_libraries)

                def build_solution(signed_libs):
                    scanned_books = set()
                    scanned_per_lib = {}
                    used_libs = []

                    current_day = 0
                    for lib in signed_libs:
                        # Validate library ID
                        if lib < 0 or lib >= self.instance.num_libs:
                            print(f"Warning: Invalid library id {lib} found in signed libraries. Skipping.")
                            continue

                        lib_data = self.instance.libs[lib]

                        if current_day + lib_data.signup_days > self.instance.num_days:
                            continue

                        current_day += lib_data.signup_days
                        remaining_days = self.instance.num_days - current_day
                        max_books = remaining_days * lib_data.books_per_day

                        # Validate books in this library
                        invalid_books = [b.id for b in lib_data.books if b.id < 0 or b.id >= self.instance.num_books]
                        if invalid_books:
                            print(f"Warning: Library {lib} has invalid book IDs: {invalid_books}")

                        # Filter only valid book IDs and exclude already scanned ones
                        available_books = [
                            b.id for b in lib_data.books
                            if b.id not in scanned_books and 0 <= b.id < self.instance.num_books
                        ]

                        # Sort by score and pick top books
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

                return (build_solution(offspring1_signed), build_solution(offspring2_signed))

            except Exception as e:
                print(f"Crossover failed: {e}, returning parents")
                return parent1, parent2
