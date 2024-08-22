# plotting.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap_with_points(func, bounds=[-100, 100], cases=None, name=""):
    """
    Generates and plots a heatmap of the function, and overlays specified points.

    Parameters:
    - func: function, The function to be plotted.
    - bounds: list, The bounds for the grid (default is [-100, 100]).
    - cases: object, Contains the cases to evaluate and plot points based on minimum fitness.

    Returns:
    None
    """
    # Generate a grid of x and y values
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the function at each point on the grid
    Z = np.array([func.evaluate([xi, yi]) for xi, yi in zip(X.flatten(), Y.flatten())])
    Z = Z.reshape(X.shape)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(Z, cmap='viridis', xticklabels=False, yticklabels=False)
    
    indices = None
    point_coords = None
    fitness = None
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
        fitness = [
            np.min(cases.randomWithOnepointFitness),
            np.min(cases.tournamentWithOnepointFitness),
            np.min(cases.randomWithMultipointsFitness),
            np.min(cases.tournamentWithMultipointsFitness),
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
    print("==========================================")
    plt.title('Function Heatmap with Points Overlay')
    plt.xlabel('x')
    plt.ylabel('y')
    print(name)
    print(f'Best Point for case Random With Onepoint: {point_coords[0]} represented by the circle')
    print(f'Best Point for case Tournment With Onepoint: {point_coords[1]} represented by the triangle')
    print(f'Best Point for case Random With MultiPoints: {point_coords[2]} represented by the square')
    print(f'Best Point for case Tournment With MultiPoints: {point_coords[3]} represented by the star')

    print(f'Best Fitness for case Random With Onepoint: {fitness[0]} represented by the circle')
    print(f'Best Fitness for case Tournment With Onepoint: {fitness[1]} represented by the triangle')
    print(f'Best Fitness for case Random With MultiPoints: {fitness[2]} represented by the square')
    print(f'Best Fitness for case Tournment With MultiPoints: {fitness[3]} represented by the star')
    print("==========================================")
    plt.show()


def plot_heatmap_with_points_testCase(func, bounds=[-100, 100], cases=None, name=""):
    """
    Generates and plots a heatmap of the function, and overlays specified points.

    Parameters:
    - func: function, The function to be plotted.
    - bounds: list, The bounds for the grid (default is [-100, 100]).
    - cases: object, Contains the cases to evaluate and plot points based on minimum fitness.

    Returns:
    None
    """
    # Generate a grid of x and y values
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the function at each point on the grid
    Z = np.array([func.evaluate([xi, yi]) for xi, yi in zip(X.flatten(), Y.flatten())])
    Z = Z.reshape(X.shape)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(Z, cmap='viridis', xticklabels=False, yticklabels=False)
    probCrossVect = [0.7,0.75,0.8,0.85,0.9,0.95]
    k_wayTournamentVect = [0.2,0.3,0.4,0.5]
    indices = None
    point_coords = None
    fitness = None
    # If specific cases are provided, find the indices of their minimum values
    if cases is not None:
        indices = np.argmin(cases.tournamentTestCaseFitness)

        point_coords = cases.tournamentTestCaseResult[indices]

        fitness = np.min(cases.tournamentTestCaseFitness)
        
    # Plot points if coordinates are provided
    if point_coords is not None:
        point_x, point_y = point_coords
        # Find the closest indices in the grid to plot the point
        ix = np.argmin(np.abs(x - point_x))
        iy = np.argmin(np.abs(y - point_y))
        
        plt.plot(ix + 0.5, iy + 0.5, 'o', color='red' , markersize=10)  
            

    # Add labels and title
    print("==========================================")
    plt.title('Function Heatmap with Points Overlay')
    plt.xlabel('x')
    plt.ylabel('y')
    print(name)
    print(f"In this cenario of comparation of {probCrossVect} Crossover probablitiry values and {k_wayTournamentVect} values for k in tournment")
    print(f'Best Point for case All test Cases: {point_coords} represented by the circle')
    print(f'Best Fitness for case All test Cases: {fitness} represented by the circle')
    print(f'The Best Crossover probability: {probCrossVect[indices//4]}')
    print(f'The Best K value for Tournament: {k_wayTournamentVect[indices%4]}')
    print()
    print("==========================================")
    plt.show()
    
