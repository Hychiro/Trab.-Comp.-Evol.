import numpy as np
import mealpy as mp
from mealpy import FloatVar


from opfunu.cec_based.cec2014 import F12014, F112014



funcConvex = F12014(ndim=10)
funcNConvex = F112014(ndim=10)

problem_dictConvex = {
    "bounds": FloatVar(lb=funcConvex.lb, ub=funcConvex.ub),
    "obj_func": funcConvex.evaluate,
    "minmax": "min",
}   

problem_dictNConvex = {
    "bounds": FloatVar(lb=funcNConvex.lb, ub=funcNConvex.ub),
    "obj_func": funcNConvex.evaluate,
    "minmax": "min",
}   

term_dict = {
   "max_early_stop": 50  # after 30 epochs, if the global best doesn't improve then we stop the program
}
    
class TrabCases:
    def __init__(self):
        self.GA = mp.evolutionary_based.GA
        pass
    def randomWithOnepoint(self, problem_dict, term):
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, selection="random", crossover="one_point")
        result = model.solve(problem_dict,termination=term)
        print(f"Case with Random selection and One point Crossover")
        print(f"Solution: {result.solution}, Fitness: {result.target.fitness}")
        print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
        
    def tournamentWithOnepoint(self,problem_dict, term):
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, selection="tournament", crossover="one_point")
        result = model.solve(problem_dict,termination=term)
        print(f"Case with Tournament selection and One point Crossover")
        print(f"Solution: {result.solution}, Fitness: {result.target.fitness}")
        print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    def randomWithMultipoints(self,problem_dict, term):
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, selection="random", crossover="multi_points")
        result = model.solve(problem_dict,termination=term)
        print(f"Case with Random selection and Multi point Crossover")
        print(f"Solution: {result.solution}, Fitness: {result.target.fitness}")
        print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    def tournamentWithMultipoints(self,problem_dict, term):
        model = self.GA.EliteMultiGA(epoch=1000, pop_size=50, selection="tournament", crossover="multi_points")
        result = model.solve(problem_dict,termination=term)
        print(f"Case with Tournament selection and Multi point Crossover")
        print(f"Solution: {result.solution}, Fitness: {result.target.fitness}")
        print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    # def randomWithMultipoints2(self):
    #     result = self.GA.EliteMultiGA(selection="random", crossover="multi_points")
    #     pass
    # def tournamentWithMultipoints2(self):
    #     result = self.GA.EliteMultiGA(selection="tournament", crossover="multi_points")
    #     pass
    # def randomWithMultipoints2(self):
    #     result = self.GA.EliteMultiGA(selection="random", crossover="multi_points")
    #     pass
    # def tournamentWithMultipoints2(self):
    #     result = self.GA.EliteMultiGA(selection="tournament", crossover="multi_points")
    #     pass

cases = TrabCases()
cases.randomWithOnepoint(problem_dict=problem_dictConvex,term=term_dict)
cases.tournamentWithOnepoint(problem_dict=problem_dictConvex,term=term_dict)
cases.randomWithMultipoints(problem_dict=problem_dictConvex,term=term_dict)
cases.tournamentWithMultipoints(problem_dict=problem_dictConvex,term=term_dict)

cases.randomWithOnepoint(problem_dict=problem_dictNConvex,term=term_dict)
cases.tournamentWithOnepoint(problem_dict=problem_dictNConvex,term=term_dict)
cases.randomWithMultipoints(problem_dict=problem_dictNConvex,term=term_dict)
cases.tournamentWithMultipoints(problem_dict=problem_dictNConvex,term=term_dict)