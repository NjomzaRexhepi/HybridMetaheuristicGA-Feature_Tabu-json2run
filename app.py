from models import Parser
from models import Solver

import os

from models.genetic_solver import GeneticAlgorithmSolver

solver = Solver()

directory = os.listdir('input')

files = ['b_read_on.txt',
         'c_incunabula.txt',
         'd_tough_choices.txt',
         'e_so_many_books.txt',
         'f_libraries_of_the_world.txt',
         'B5000_L90_D21.txt',
         'B50000_L400_D28.txt',
         'B90000_L850_D21.txt',
         'B95000_L2000_D28.txt']

for file in directory:
    if files.__contains__(file):
        print(f'Computing ./input/{file}')
        parser = Parser(f'./input/{file}')
        instance = parser.parse()
        initial_solution = solver.generate_initial_solution_grasp(instance, max_time=20)
        geneticSolver = GeneticAlgorithmSolver(instance, initial_solution)
        solution = geneticSolver.solve()
        solution.export(f'./output/{file}')
        print(f"{solution.fitness_score:,}", file)

