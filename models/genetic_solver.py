import random
from typing import List, Tuple
 
import json
from models import InstanceData, Solution
from models.solver import Solver
import sys
import heapq
from typing import Tuple
 
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
 

        # TODO: Uncomment if you want to use neps and GPU
        #  def __init__(self, instance: InstanceData, initial_solution: Solution,
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
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def load(self, file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
 
        def generate_initial_solution(self, instance):
            """Generate initial solution using GRASP"""
            return self.solver.generate_initial_solution_grasp(instance, max_time=30)
    
        def solve(self) -> Solution:
            population = self.initialize_population(self.initial_solution)
            best_solution = max(population, key=lambda x: x.fitness_score)
            best_score = best_solution.fitness_score

            no_improvement_counter = 0
            patience = 10  # adjustable

            for generation in range(self.population_size):
                print(f"Gen {generation}: Best fitness = {best_score}")

                new_population = [best_solution]  # Elitism

                while len(new_population) < self.population_size:
                    parent1 = self.tournament_select(population)
                    parent2 = self.tournament_select(population)

                    offspring1, offspring2 = self.union_crossover(parent1, parent2)

                    for offspring in (offspring1, offspring2):
                        if random.random() < self.mutation_prob:
                            offspring = self.solver.feature_based_tabu_search(
                                offspring, self.instance, max_iterations=self.hill_climbing_steps
                            )
                        new_population.append(offspring)

                population = new_population[:self.population_size]
                population.sort(key=lambda x: x.fitness_score)
                current_best = population[0]

                if current_best is not best_solution:
                    best_solution = current_best
                    best_score = current_best.fitness_score
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

                if no_improvement_counter >= patience:
                    print(f"Early stopping: No improvement for {patience} generations.")
                    break

            return best_solution
    
        def initialize_population(self, initial_solution: Solution) -> List[Solution]:
            """Create initial population with variations of the initial solution"""
            population = [self.initial_solution]
    
            while len(population) < self.population_size:
                variant_fitness, variant = self.solver.hill_climbing_combined_w_initial_solution(initial_solution, self.instance, iterations=5)
                population.append(variant)
    
            return population
        
        def tournament_select(self, population: List[Solution]) -> Solution:
            """Select best solution out of random tournament_size candidates"""
            tournament = random.sample(population, self.tournament_size)
            return max(tournament, key=lambda x: x.fitness_score)

        @staticmethod
        def book_value(book_id, scores, book_rarity):
            rarity = book_rarity[book_id]
            return scores[book_id] / rarity if rarity > 0 else scores[book_id]

        # @staticmethod
        # def build_solution(signed_libs, scores, num_books, num_days, libs_data, total_libs_set, book_rarity):
        #     scanned_books = set()
        #     scanned_per_lib = {}
        #     used_libs = []
        #     current_day = 0

        #     for lib in signed_libs:
        #         if not (0 <= lib < len(libs_data)):
        #             continue

        #         lib_data = libs_data[lib]
        #         signup_days = lib_data.signup_days

        #         if current_day + signup_days >= num_days:
        #             break  # Early termination

        #         current_day += signup_days
        #         remaining_days = num_days - current_day
        #         max_books = remaining_days * lib_data.books_per_day

        #         available_books = [
        #             b.id for b in lib_data.books
        #             if b.id not in scanned_books and 0 <= b.id < num_books
        #         ]

        #         if available_books:
        #             selected = heapq.nlargest(
        #                 max_books,
        #                 available_books,
        #                 key=lambda x: GeneticAlgorithmSolver.book_value(x, scores, book_rarity)
        #             )
        #             scanned_books.update(selected)
        #             scanned_per_lib[lib] = selected
        #             used_libs.append(lib)

        #     return Solution(
        #         signed_libs=used_libs,
        #         unsigned_libs=list(total_libs_set - set(used_libs)),
        #         scanned_books_per_library=scanned_per_lib,
        #         scanned_books=scanned_books
        #     )

        # def union_crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        #     try:
        #         all_libs = list(dict.fromkeys(parent1.signed_libraries + parent2.signed_libraries))
        #         total_libs_set = set(range(self.instance.num_libs))
        #         scores = self.instance.scores
        #         num_books = self.instance.num_books
        #         num_days = self.instance.num_days
        #         libs_data = self.instance.libs

        #         # Compute book rarity
        #         book_rarity = [0] * num_books
        #         for lib in libs_data:
        #             for book in lib.books:
        #                 book_rarity[book.id] += 1

        #         def library_heuristic(lib_id):
        #             lib_data = libs_data[lib_id]
        #             potential_books = [b.id for b in lib_data.books]
        #             value = sum(GeneticAlgorithmSolver.book_value(b, scores, book_rarity) for b in potential_books)
        #             return value / lib_data.signup_days if lib_data.signup_days > 0 else value

        #         sorted_libs = sorted(all_libs, key=library_heuristic, reverse=True)

        #         offspring1 = GeneticAlgorithmSolver.build_solution(sorted_libs, scores, num_books, num_days, libs_data, total_libs_set, book_rarity)
        #         offspring2 = GeneticAlgorithmSolver.build_solution(list(reversed(sorted_libs)), scores, num_books, num_days, libs_data, total_libs_set, book_rarity)

        #         offspring1.calculate_fitness_score(scores)
        #         offspring2.calculate_fitness_score(scores)

        #         return offspring1, offspring2

        #     except Exception as e:
        #         print(f"Union crossover failed: {e}, returning parents")
        #         parent1.calculate_fitness_score(self.instance.scores)
        #         parent2.calculate_fitness_score(self.instance.scores)
        #         return parent1, parent2
