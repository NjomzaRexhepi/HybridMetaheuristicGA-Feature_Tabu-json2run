from models import Parser
from models import Solver

import os

from models.genetic_solver import GeneticAlgorithmSolver

solver = Solver()

directory = os.listdir('input')

files = [
    'b_read_on.txt',
    'c_incunabula.txt',
    'e_so_many_books.txt',
    'd_tough_choices.txt',
    'f_libraries_of_the_world.txt',
    'B50_L5_D4.txt',
    'B60_L6_D6.txt',
    'B70_L8_D7.txt',
    'B80_L7_D9.txt',
    'B90_L6_D11.txt',
    'B95_L5_D12.txt',
    'B96_L6_D14.txt',
    'B98_L4_D15.txt',
    'B99_L7_D16.txt',
    'B100_L9_D18.txt',
    'B150_L8_D10.txt',
    'B300_L11_D20.txt',
    'B500_L14_D18.txt',
    'B750_L20_D14.txt',
    'B1k_L18_D17.txt',
    'B1.5k_L20_D40.txt',
    'B2.5k_L25_D50.txt',
    'B4k_L30_D60.txt',
    'B5k_L90_D21.txt',
    'B6k_L35_D70.txt',
    'B9k_L40_D80.txt',
    'B12k_L45_D90.txt',
    'B18k_L50_D100.txt',
    'B25k_L55_D110.txt',
    'B35k_L60_D120.txt',
    'B48k_L65_D130.txt',
    'B50k_L400_D28.txt',
    'B60k_L70_D140.txt',
    'B80k_L75_D150.txt',
    'B90k_L850_D21.txt',
    'B95k_L2k_D28.txt',
    'B100k_L600_D28.txt',
    'B120k_L80_D160.txt',
    'B180k_L85_D170.txt',
    'B240k_L90_D180.txt',
    'B300k_L95_D190.txt',
    'B450k_L100_D200.txt',
    'B600k_L105_D210.txt',
    'B800k_L110_D220.txt',
    'B1000k_L115_D230.txt',
    'B1200k_L120_D240.txt',
    'B1400k_L125_D250.txt',
    'B1600k_L130_D260.txt',
    'B1800k_L135_D270.txt',
    'B2000k_L140_D280.txt',
    'B2.5k_L3_D90 (Appalachian Regional Project).txt',
    'B18k_L4_D365 (Oxford Bodleian Archives).txt',
    'B150k_L92_730D (New York Public Library System).txt',
    'B2000k_L300_D3.6k (Deutsche Digitale Bibliothek).txt',
    'B600k_L40_D540 (Internet Archive Bulk Scanning).txt',
    'B3k_L1_D1.4k (Vatican Secret Archives Project).txt'
         ]

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

