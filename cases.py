# cases.py

import numpy as np
import mealpy as mp

class TrabCases:
    
    def __init__(self):
        self.GA = mp.evolutionary_based.GA
        self.CSA = mp.swarm_based.CSA
        self.randomWithOnepointResult = []
        self.randomWithOnepointFitness = []
        self.randomWithOnepointModels = []
        self.tournamentWithOnepointResult = []
        self.tournamentWithOnepointFitness = []
        self.tournamentWithOnepointModels = []
        self.randomWithMultipointsResult = []
        self.randomWithMultipointsFitness = []
        self.randomWithMultipointsModels = []
        self.tournamentWithMultipointsResult = []
        self.tournamentWithMultipointsFitness = []
        self.tournamentWithMultipointsModels = []
        self.tournamentTestCaseResult = []
        self.tournamentTestCaseFitness = []
        self.tournamentTestCaseModels = []
        self.csaCaseResult = []
        self.csaCaseFitness = []
        self.csaCaseModels =[]
    def npArray(self):
        self.randomWithOnepointResult = np.array(self.randomWithOnepointResult)
        self.randomWithOnepointFitness = np.array(self.randomWithOnepointFitness)
        self.randomWithOnepointModels = np.array(self.randomWithOnepointModels)

        self.tournamentWithOnepointResult = np.array(self.tournamentWithOnepointResult)
        self.tournamentWithOnepointFitness = np.array(self.tournamentWithOnepointFitness)
        self.tournamentWithOnepointModels = np.array(self.tournamentWithOnepointModels)

        self.randomWithMultipointsResult = np.array(self.randomWithMultipointsResult)
        self.randomWithMultipointsFitness = np.array(self.randomWithMultipointsFitness)
        self.randomWithMultipointsModels = np.array(self.randomWithMultipointsModels)

        self.tournamentWithMultipointsResult = np.array(self.tournamentWithMultipointsResult)
        self.tournamentWithMultipointsFitness = np.array(self.tournamentWithMultipointsFitness)
        self.tournamentWithMultipointsModels = np.array(self.tournamentWithMultipointsModels)
        
        self.tournamentTestCaseResult = np.array(self.tournamentTestCaseResult)
        self.tournamentTestCaseFitness = np.array(self.tournamentTestCaseFitness)
        self.tournamentTestCaseModels = np.array(self.tournamentTestCaseModels)

        self.csaCaseResult = np.array(self.csaCaseResult)
        self.csaCaseFitness = np.array(self.csaCaseFitness)
        self.csaCaseModels =np.array(self.csaCaseModels)

    def randomWithOnepoint(self, problem_dict, term=None):
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, selection="random", crossover="multi_points")
        result = model.solve(problem_dict, termination=term)
        self.randomWithOnepointResult.append(result.solution)
        self.randomWithOnepointFitness.append(result.target.fitness)
        self.randomWithOnepointModels.append(model)
        
        
        
    def tournamentWithOnepoint(self, problem_dict, term=None):
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, selection="tournament", crossover="one_point")
        result = model.solve(problem_dict, termination=term)
        self.tournamentWithOnepointResult.append(result.solution)
        self.tournamentWithOnepointFitness.append(result.target.fitness)
        self.tournamentWithOnepointModels.append(model)

    def randomWithMultipoints(self, problem_dict, term=None):
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, selection="random", crossover="multi_points")
        result = model.solve(problem_dict, termination=term)
        self.randomWithMultipointsResult.append(result.solution)
        self.randomWithMultipointsFitness.append(result.target.fitness)
        self.randomWithMultipointsModels.append(model)

    def tournamentWithMultipoints(self, problem_dict, term=None):
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, selection="tournament", crossover="multi_points")
        result = model.solve(problem_dict, termination=term)
        self.tournamentWithMultipointsResult.append(result.solution)
        self.tournamentWithMultipointsFitness.append(result.target.fitness)
        self.tournamentWithMultipointsModels.append(model)

    def testCaseTournament(self, problem_dict, term=None, pc = 0.95, k_way = 0.2):
        #pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        #k_way (float): Optional, set it when use “tournament” selection, default = 0.2
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, pc=pc, k_way=k_way, selection="tournament", crossover="multi_points")
        result = model.solve(problem_dict, termination=term)
        self.tournamentTestCaseResult.append(result.solution)
        self.tournamentTestCaseFitness.append(result.target.fitness)
        self.tournamentTestCaseModels.append(model)

    def csaCase(self, problem_dict, term=None, p_a=0.3):
        model = self.CSA.OriginalCSA(epoch=1000, pop_size=50,p_a=p_a)
        result = model.solve(problem_dict, termination=term)
        self.csaCaseResult.append(result.solution)
        self.csaCaseFitness.append(result.target.fitness)
        self.csaCaseModels.append(model)
