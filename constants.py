
import numpy as np
import mealpy as mp
from mealpy import FloatVar
import opfunu
from opfunu.cec_based.cec2014 import F12014, F112014, F42014,F82014

# Define dimensions and bounds
DIM = 2
BOUNDS = [-100, 100]

# Initialize functions
FUNC_CONVEX = F12014(ndim=DIM)
FUNC_NCONVEX = F112014(ndim=DIM)

# Problem dictionaries
PROBLEM_DICT_CONVEX = {
    "bounds": FloatVar(lb=FUNC_CONVEX.lb * DIM, ub=FUNC_CONVEX.ub * DIM),
    "obj_func": FUNC_CONVEX.evaluate,
    "minmax": "min",
    "save_population": True
}

PROBLEM_DICT_NCONVEX = {
    "bounds": FloatVar(lb=FUNC_NCONVEX.lb * DIM, ub=FUNC_NCONVEX.ub * DIM),
    "obj_func": FUNC_NCONVEX.evaluate,
    "minmax": "min",
    "save_population": True
}

DIM = 10
# Initialize functions
FUNC_CONVEX10= F12014(ndim=DIM)
FUNC_NCONVEX10 = F112014(ndim=DIM)
NAME_CONVEX = F12014.name
NAME_NCONVEX = F112014.name

# Problem dictionaries
PROBLEM_DICT_CONVEX10 = {
    "bounds": FloatVar(lb=FUNC_CONVEX10.lb * DIM, ub=FUNC_CONVEX10.ub * DIM),
    "obj_func": FUNC_CONVEX10.evaluate,
    "minmax": "min",
    "save_population": True
}

PROBLEM_DICT_NCONVEX10 = {
    "bounds": FloatVar(lb=FUNC_NCONVEX10.lb * DIM, ub=FUNC_NCONVEX10.ub * DIM),
    "obj_func": FUNC_NCONVEX10.evaluate,
    "minmax": "min",
    "save_population": True
}

# Termination dictionary
TERM_DICT = {
    "max_early_stop": 150
}


DIM = 2
FUNC_5= F42014(ndim=DIM)
FUNC_9 = F82014(ndim=DIM)
NAME_5 = "Rosenbrock"
NAME_9 = "Rastrigin"


PROBLEM_DICT_5 = {
    "bounds": FloatVar(lb=FUNC_5.lb * DIM, ub=FUNC_5.ub * DIM),
    "obj_func": FUNC_5.evaluate,
    "minmax": "min",
    "save_population": True
}

PROBLEM_DICT_9 = {
    "bounds": FloatVar(lb=FUNC_9.lb * DIM, ub=FUNC_9.ub * DIM),
    "obj_func": FUNC_9.evaluate,
    "minmax": "min",
    "save_population": True
}