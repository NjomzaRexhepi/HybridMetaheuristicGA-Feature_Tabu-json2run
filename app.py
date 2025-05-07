from models import Parser
from models import Solver

import os

from models.genetic_solver import GeneticAlgorithmSolver

solver = Solver()

directory = os.listdir('input')

files = ['b_read_on.txt',
         'c_incunabula.txt',
         'e_so_many_books.txt',
         'd_tough_choices.txt',
         'f_libraries_of_the_world.txt',
         'B50_L5_D4.in',
         'B60_L6_D6.in',
         'B70_L8_D7.in',
         'B80_L7_D9.in',
         'B90_L6_D11.in',
         'B95_L5_D12.in',
         'B96_L6_D14.in',
         'B98_L4_D15.in',
         'B99_L7_D16.in',
         'B100_L9_D18.in',
         'B150_L8_D10.in',
         'B300_L11_D20.in',
         'B500_L14_D18.in',
         'B750_L20_D14.in',
         'B1k_L18_D17.in',
         'B1.5k_L20_D40.in',
         'B2.5k_L25_D50.in',
         'B4k_L30_D60.in',
         'B5k_L90_D21.txt',
         'B6k_L35_D70.in',
         'B9k_L40_D80.in',
         'B12k_L45_D90.in',
         'B18k_L50_D100.in',
         'B25k_L55_D110.in',
         'B35k_L60_D120.in',
         'B48k_L65_D130.in',
         'B50k_L400_D28.txt',
         'B60k_L70_D140.in',
         'B80k_L75_D150.in',
         'B90k_L850_D21.txt',
         'B95k_L2k_D28.txt',
         'B100k_L600_D28.txt',
         'B120k_L80_D160.in',
         'B180k_L85_D170.in',
         'B240k_L90_D180.in',
         'B300k_L95_D190.in',
         'B450k_L100_D200.in',
         'B600k_L105_D210.in',
         'B800k_L110_D220.in',
         'B1000k_L115_D230.in',
         'B1200k_L120_D240.in',
         'B1400k_L125_D250.in',
         'B1600k_L130_D260.in',
         'B1800k_L135_D270.in',
         'B2000k_L140_D280.in',
         'B2.5k_L3_D90 (Appalachian Regional Project).in',
         'B18k_L4_D365 (Oxford Bodleian Archives).in',
         'B150k_L92_730D (New York Public Library System).in',
         'B2000k_L300_D3.6k (Deutsche Digitale Bibliothek).in',
         'B600k_L40_D540 (Internet Archive Bulk Scanning).in',
         'B3k_L1_D1.4k (Vatican Secret Archives Project).in'
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

