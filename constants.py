
import numpy as np
import mealpy as mp
from mealpy import FloatVar
import opfunu
from opfunu.cec_based.cec2014 import F12014, F112014

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