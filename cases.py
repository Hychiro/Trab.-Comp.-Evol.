# cases.py

import numpy as np
import mealpy as mp

class TrabCases:
    
    def __init__(self):
        self.GA = mp.evolutionary_based.GA
        self.randomWithOnepointResult = []
        self.randomWithOnepointFitness = []
        self.tournamentWithOnepointResult = []
        self.tournamentWithOnepointFitness = []
        self.randomWithMultipointsResult = []
        self.randomWithMultipointsFitness = []
        self.tournamentWithMultipointsResult = []
        self.tournamentWithMultipointsFitness = []
        self.tournamentTestCaseResult = []
        self.tournamentTestCaseFitness = []
    
    def npArray(self):
        self.randomWithOnepointResult = np.array(self.randomWithOnepointResult)
        self.randomWithOnepointFitness = np.array(self.randomWithOnepointFitness)
        self.tournamentWithOnepointResult = np.array(self.tournamentWithOnepointResult)
        self.tournamentWithOnepointFitness = np.array(self.tournamentWithOnepointFitness)
        self.randomWithMultipointsResult = np.array(self.randomWithMultipointsResult)
        self.randomWithMultipointsFitness = np.array(self.randomWithMultipointsFitness)
        self.tournamentWithMultipointsResult = np.array(self.tournamentWithMultipointsResult)
        self.tournamentWithMultipointsFitness = np.array(self.tournamentWithMultipointsFitness)
        self.tournamentTestCaseResult = np.array(self.tournamentTestCaseResult)
        self.tournamentTestCaseFitness = np.array(self.tournamentTestCaseFitness)

    def randomWithOnepoint(self, problem_dict, term=None):
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, selection="random", crossover="multi_points")
        result = model.solve(problem_dict, termination=term)
        self.randomWithOnepointResult.append(result.solution)
        self.randomWithOnepointFitness.append(result.target.fitness)
        
    def tournamentWithOnepoint(self, problem_dict, term=None):
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, selection="tournament", crossover="one_point")
        result = model.solve(problem_dict, termination=term)
        self.tournamentWithOnepointResult.append(result.solution)
        self.tournamentWithOnepointFitness.append(result.target.fitness)

    def randomWithMultipoints(self, problem_dict, term=None):
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, selection="random", crossover="multi_points")
        result = model.solve(problem_dict, termination=term)
        self.randomWithMultipointsResult.append(result.solution)
        self.randomWithMultipointsFitness.append(result.target.fitness)

    def tournamentWithMultipoints(self, problem_dict, term=None):
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, selection="tournament", crossover="multi_points")
        result = model.solve(problem_dict, termination=term)
        self.tournamentWithMultipointsResult.append(result.solution)
        self.tournamentWithMultipointsFitness.append(result.target.fitness)

    def testCaseTournament(self, problem_dict, term=None, pc = 0.95, k_way = 0.2):
        #pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        #k_way (float): Optional, set it when use “tournament” selection, default = 0.2
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, pc=pc, k_way=k_way, selection="tournament", crossover="multi_points")
        result = model.solve(problem_dict, termination=term)
        self.tournamentTestCaseResult.append(result.solution)
        self.tournamentTestCaseFitness.append(result.target.fitness)
