import numpy as np
import mealpy as mp
from mealpy import FloatVar, optimizer
import opfunu
from opfunu.cec_based.cec2014 import F12014, F112014
import matplotlib.pyplot as plt
import seaborn as sns


dim = 2
funcConvex = F12014(ndim=dim)
funcNConvex = F112014(ndim=dim)

# print(funcNConvex.lb, funcNConvex.ub)
problem_dictConvex = {
    "bounds": FloatVar(lb=funcConvex.lb* dim, ub=funcConvex.ub* dim),
    "obj_func": funcConvex.evaluate,
    "minmax": "min",
}   

problem_dictNConvex = {
    "bounds": FloatVar(lb=funcNConvex.lb*dim, ub=funcNConvex.ub*dim),
    "obj_func": funcNConvex.evaluate,
    "minmax": "min",
}   

term_dict = {
   "max_early_stop": 50  # after 30 epochs, if the global best doesn't improve then we stop the program
}
    
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
    
    def npArray(self):
        self.randomWithOnepointResult = np.array(self.randomWithOnepointResult)
        self.randomWithOnepointFitness = np.array(self.randomWithOnepointFitness)
        self.tournamentWithOnepointResult = np.array(self.tournamentWithOnepointResult)
        self.tournamentWithOnepointFitness = np.array(self.tournamentWithOnepointFitness)
        self.randomWithMultipointsResult = np.array(self.randomWithMultipointsResult)
        self.randomWithMultipointsFitness = np.array(self.randomWithMultipointsFitness)
        self.tournamentWithMultipointsResult = np.array(self.tournamentWithMultipointsResult)
        self.tournamentWithMultipointsFitness = np.array(self.tournamentWithMultipointsFitness)
        
    def randomWithOnepoint(self, problem_dict, term = None):
        model = self.GA.EliteMultiGA(epoch=100, pop_size=50, selection="random", crossover="multi_points")
        result = model.solve(problem_dict,termination=term)
        print(f"Case with Random selection and One point Crossover")
        print(f"Solution: {result.solution}, Fitness: {result.target.fitness}")
        self.randomWithOnepointResult.append(result.solution)
        self.randomWithOnepointFitness.append(result.target.fitness)
        
    def tournamentWithOnepoint(self,problem_dict, term = None):
        model = self.GA.EliteMultiGA(epoch=100, pop_size=50, selection="tournament", crossover="one_point")
        result = model.solve(problem_dict,termination=term)
        print(f"Case with Tournament selection and One point Crossover")
        print(f"Solution: {result.solution}, Fitness: {result.target.fitness}")
        self.tournamentWithOnepointResult.append(result.solution)
        self.tournamentWithOnepointFitness.append(result.target.fitness)

    def randomWithMultipoints(self,problem_dict, term = None):
        model = self.GA.EliteMultiGA(epoch=100, pop_size=50, selection="random", crossover="multi_points")
        result = model.solve(problem_dict,termination=term)
        print(f"Case with Random selection and Multi point Crossover")
        print(f"Solution: {result.solution}, Fitness: {result.target.fitness}")
        self.randomWithMultipointsResult.append(result.solution)
        self.randomWithMultipointsFitness.append(result.target.fitness)

    def tournamentWithMultipoints(self,problem_dict, term = None):
        model = self.GA.EliteMultiGA(epoch=100, pop_size=50, selection="tournament", crossover="multi_points")
        result = model.solve(problem_dict,termination=term)
        print(f"Case with Tournament selection and Multi point Crossover")
        print(f"Solution: {result.solution}, Fitness: {result.target.fitness}")
        self.tournamentWithMultipointsResult.append(result.solution)
        self.tournamentWithMultipointsFitness.append(result.target.fitness)

cases10dim1 = TrabCases()

for i in range(10):

  cases10dim1.randomWithOnepoint(problem_dict=problem_dictConvex)
  cases10dim1.tournamentWithOnepoint(problem_dict=problem_dictConvex)
  cases10dim1.randomWithMultipoints(problem_dict=problem_dictConvex)
  cases10dim1.tournamentWithMultipoints(problem_dict=problem_dictConvex)

cases10dim1.npArray()

cases10dim2 = TrabCases()

for i in range(10):

  cases10dim2.randomWithOnepoint(problem_dict=problem_dictNConvex)
  cases10dim2.tournamentWithOnepoint(problem_dict=problem_dictNConvex)
  cases10dim2.randomWithMultipoints(problem_dict=problem_dictNConvex)
  cases10dim2.tournamentWithMultipoints(problem_dict=problem_dictNConvex)

cases10dim2.npArray()
cases2dim1 = TrabCases()

for i in range(10):

  cases2dim1.randomWithOnepoint(problem_dict=problem_dictConvex)
  cases2dim1.tournamentWithOnepoint(problem_dict=problem_dictConvex)
  cases2dim1.randomWithMultipoints(problem_dict=problem_dictConvex)
  cases2dim1.tournamentWithMultipoints(problem_dict=problem_dictConvex)

cases2dim1.npArray()
cases2dim2 = TrabCases()

for i in range(10):

  cases2dim2.randomWithOnepoint(problem_dict=problem_dictNConvex)
  cases2dim2.tournamentWithOnepoint(problem_dict=problem_dictNConvex)
  cases2dim2.randomWithMultipoints(problem_dict=problem_dictNConvex)
  cases2dim2.tournamentWithMultipoints(problem_dict=problem_dictNConvex)

cases2dim2.npArray()

print("============== Casos Convexo e 10 dimensões ==============")
print("Randomico com um ponto:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases10dim1.randomWithOnepointFitness)/10))

print("Tournament com um ponto:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases10dim1.tournamentWithOnepointFitness)/10))

print("Randomico com multiplos pontos:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases10dim1.randomWithMultipointsFitness)/10))

print("Tournament com multiplos pontos:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases10dim1.tournamentWithMultipointsFitness)/10))

print("============== Casos não Convexo e 10 dimensões ==============")
print("Randomico com um ponto:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases10dim2.randomWithOnepointFitness)/10))

print("Tournament com um ponto:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases10dim2.tournamentWithOnepointFitness)/10))

print("Randomico com multiplos pontos:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases10dim2.randomWithMultipointsFitness)/10))

print("Tournament com multiplos pontos:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases10dim2.tournamentWithMultipointsFitness)/10))


print("============== Casos Convexo e 2 dimensões ==============")
print("Randomico com um ponto:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases2dim1.randomWithOnepointFitness)/10))

print("Tournament com um ponto:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases2dim1.tournamentWithOnepointFitness)/10))

print("Randomico com multiplos pontos:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases2dim1.randomWithMultipointsFitness)/10))

print("Tournament com multiplos pontos:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases2dim1.tournamentWithMultipointsFitness)/10))

print("============== Casos não Convexo e 2 dimensões ==============")
print("Randomico com um ponto:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases2dim2.randomWithOnepointFitness)/10))

print("Tournament com um ponto:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases2dim2.tournamentWithOnepointFitness)/10))

print("Randomico com multiplos pontos:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases2dim2.randomWithMultipointsFitness)/10))

print("Tournament com multiplos pontos:")
print("Média da fitness de 10 iteracoes: " + str(np.sum(cases2dim2.tournamentWithMultipointsFitness)/10))


def plot_heatmap_with_points(func, bounds=[-100, 100], cases=None):
    """
    Generates and plots a heatmap of the F12014 function, and overlays specified points.

    Parameters:
    - ndim: int, Number of dimensions (default is 2).
    - bounds: list, The bounds for the grid (default is [-100, 100]).
    - point_coords: list of tuples, Points to overlay on the heatmap (default is None).
    - cases: object, Contains the cases to evaluate and plot points based on minimum fitness.

    Returns:
    None
    """
    # Initialize the function with the specified bounds and dimensions
    f = func

    # Generate a grid of x and y values
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the function at each point on the grid
    Z = np.array([f.evaluate([xi, yi]) for xi, yi in zip(X.flatten(), Y.flatten())])
    Z = Z.reshape(X.shape)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(Z, cmap='viridis', xticklabels=False, yticklabels=False)
    indices = None
    point_coords = None
    # If specific cases are provided, find the indices of their minimum values
    if cases is not None:
        indices = [
            np.argmin(cases.randomWithOnepointFitness),
            np.argmin(cases.tournamentWithOnepointFitness),
            np.argmin(cases.randomWithMultipointsFitness),
            np.argmin(cases.tournamentWithMultipointsFitness),
        ]
        point_coords = [
            cases.randomWithOnepointResult[indices[0]],
            cases.tournamentWithOnepointResult[indices[1]],
            cases.randomWithMultipointsResult[indices[2]],
            cases.tournamentWithMultipointsResult[indices[3]],
        ]
        
    # Plot points if coordinates are provided
    if point_coords is not None:
        count = 0
        for point_x, point_y in point_coords:
            # Find the closest indices in the grid to plot the point
            ix = np.argmin(np.abs(x - point_x))
            iy = np.argmin(np.abs(y - point_y))
            if count == 0:
                plt.plot(ix + 0.5, iy + 0.5, 'o', color='red' , markersize=10)  
            elif count == 1:
                plt.plot(ix + 0.5, iy + 0.5, 'v', color='red' , markersize=10)  
            elif count  == 2:
                plt.plot(ix + 0.5, iy + 0.5, 's', color='red' , markersize=10)  
            else:
                plt.plot(ix + 0.5, iy + 0.5, '*', color='red' , markersize=10)  
            count = count + 1

    # Add labels and title
    plt.title('F12014 with Points Overlay')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Example usage:
plot_heatmap_with_points(func=funcConvex, cases=cases2dim1)
plot_heatmap_with_points(func=funcNConvex, cases=cases2dim2)
