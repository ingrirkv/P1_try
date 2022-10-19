

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import Sub_problem as Sub_problem
import Create_cuts as Create_cut


def MasterProblem(Cuts_data):  # ha med itaration,
    #This is the master problem variable output

    """set, variabler osv"""

    model = pyo.ConcreteModel()

    """sets"""
    T_1_range = 24
    model.T_1 = pyo.RangeSet(0, T_1_range)

    """parametere"""
    V_0 = 5  # starting volume in the reservoir given in Mm3 for t = 0
    V_MAX = 10  # maximum volume in the reservoir given in Mm3
    Q_Max = 0.36  # maximum outflow per hour from the reservoir given in Mm3/h
    P_MAX = 100  # maximum production per hour given in MW
    M_conv = 0.0036  # conversion factor given in [Mm3/m^3]
    E_conv = 0.981  # conversion factor for discharge water to produce electricity, [Mm3/m^3]
    I_1 = 0.18  # Inflow during the first 24 hours


    # initialize the parameters to the model:
    model.V_0 = pyo.Param(initialize=V_0)
    model.V_MAX = pyo.Param(initialize=V_MAX)
    model.Q_Max = pyo.Param(initialize=Q_Max)
    model.P_Max = pyo.Param(initialize=P_MAX)
    model.M_conv = pyo.Param(initialize=M_conv)
    model.E_conv = pyo.Param(initialize=E_conv)
    model.I_1 = pyo.Param(initialize=I_1)


    # price:
    Dict = {}
    for t in range(25):
        Dict[t] = 50 + t
    model.p_t = pyo.Param(model.T_1, initialize=Dict)
    print("Pris", Dict)



    "Variables"
    model.P_t = pyo.Var(model.T_1, domain=pyo.NonNegativeReals)  # produced electricity
    model.Q_t = pyo.Var(model.T_1, domain=pyo.NonNegativeReals)  # outflow
    model.V1_t = pyo.Var(model.T_1, domain=pyo.NonNegativeReals)  # volume of water during the first 24 hours
    #model.b = pyo.Var(model.Iteration, domain=pyo.NonNegativeReals)  # b i y = ax+b
    #model.dual = pyo.Var(model.Iteration, domain=pyo.NonNegativeReals)  # a
    model.x_1 = pyo.Var(model.Iteration, domain=pyo.NonNegativeReals)
    model.alpha = pyo.Var(model.Iteration, domain=pyo.NonNegativeReals, bounds=(-10e6, 10e6))  # usikker på om alpha er en variabel
    # master

    """objective function"""
    def objective_func(model):
        obj_del1 = sum(model.P_t[t] * model.p_t[t] for t in model.T_1) + model.alpha  # Cuts_data.   # Objective function, er noe feil her for vi må egentlig bruke den Cuts_data
        return obj_del1
    model.obj_del1 = pyo.Objective(rule=objective_func, sense=pyo.maximize)


    """constraints for master problem"""
    # constraint 1, ensure that Q_t is lower than Qmax
    def constraint_Q(model, t):
        if t == 0:
            return model.Q_t[t] == 0
        else:
            return model.Q_t[t] <= model.Q_Max

    model.C1 = pyo.Constraint(model.T_1, rule=constraint_Q)

    # constraint 2, constraint for water volume
    def constraint_V(model, t):
        if (t == 0):
            return (model.V1_t[0] == 5)
        else:
            return (model.V1_t[t] == model.V1_t[t - 1] + model.I_1 - model.Q_t[t])

    model.C2 = pyo.Constraint(model.T_1, rule=constraint_V)

    # Constraint 3, ensure that V_t is lower than Vmax
    def constraint_V2(model, t):
        return (model.V1_t[t] <= model.V_MAX)

    model.C3 = pyo.Constraint(model.T_1, rule=constraint_V2)

    # Constraint 4, P_vt, kan legges i obj
    def constraint_P1(model, t):
        return (model.P_t[t] == model.Q_t[t] * model.E_conv * (1 / model.M_conv))

    model.C4 = pyo.Constraint(model.T_1, rule=constraint_P1)

    # Constraint 5, ensure that P_ts is lower than Pmax, trenger egentlig ikke
    def constraint_Pmax(model, t):
        return (model.P_t[t] <= model.P_Max)

    model.C5 = pyo.Constraint(model.T_1, rule=constraint_Pmax)

# en liste med cut
    List_of_cuts = []
    for el in Cuts_data:
        List_of_cuts.append(el)

    model.Cuts = pyo.Set(initialize=List_of_cuts)  # tell us how many cust we have in the model
    model.Cuts_data = Cuts_data # lager set med hvor mange cut vi har

    def Constraint_cuts(model, cut):
        print(model.Cuts_data[cut]["slope"], model.Cuts_data[cut]["constant"])
        print("Creating cut: ", cut)
        model.aplha <= model.Cuts_data[cut]("slope") * model.V1_t[24] + model.Cuts_data[cut]("constant")
        return (model.alpha)
    model.C5 = pyo.Constraint(model.Cuts, rule=Constraint_cuts)


    # vi løser den her med solver osv
    solver = 'gurobi'
    opt = SolverFactory(solver, load_solution=True)
    results = opt.solve(model, load_solutions=True)
    print("result", results)
    model.display()
    model.x_1 = model.V1_t[24]

    return model.x_1
