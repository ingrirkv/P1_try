import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import Master_Problem as mp



def SubProblem(x_1):
    model_2 = pyo.ConcreteModel()

    """sets"""
    T_2_range = 48
    model_2.T_2 = pyo.RangeSet(0,T_2_range)

    """parametere"""
    V_MAX = 10             #maximum volume in the reservoir given in Mm3
    Q_Max = 0.36           #maximum outflow per hour from the reservoir given in Mm3/h
    P_MAX = 100            #maximum production per hour given in MW
    M_conv = 0.0036        #conversion factor given in [Mm3/m^3]
    E_conv = 0.981         #conversion factor for discharge water to produce electricity, [Mm3/m^3]
    WV_end = 13000         #end water value for all scenarios given in EUR/Mm3
    I_2 = 0.18             #Inflow during the second 24 hours


    #initialize the parameters to the model:

    model_2.V_MAX = pyo.Param(initialize=V_MAX)
    model_2.Q_Max = pyo.Param(initialize=Q_Max)
    model_2.P_Max = pyo.Param(initialize=P_MAX)
    model_2.M_conv = pyo.Param(initialize=M_conv)
    model_2.E_conv = pyo.Param(initialize=E_conv)
    model_2.WV_end = pyo.Param(initialize=WV_end)
    model_2.I_2 = pyo.Param(initialize=I_2)


    #price:
    Dict={}
    for t in range(25, 48):
        Dict[t]= 50+t
    model_2.p_t2 = pyo.Param(model_2.T_2, initialize=Dict)
    print("Pris", Dict)

    "Variables"
    model_2.P_t = pyo.Var(model_2.T_2, domain=pyo.NonNegativeReals) #produced electricity
    model_2.Q_t = pyo.Var(model_2.T_2, domain=pyo.NonNegativeReals) #outflow
    model_2.V1_t = pyo.Var(model_2.T_2, domain=pyo.NonNegativeReals) #volume of water


#Subproblem
    def obj_2(model_2):
        OBJ2 = sum(model_2.P_t[t] * model_2.p_t[t] for t in model_2.T2) + model_2.WV_end * model_2.V1_t[48]  # Objective function
        return OBJ2
    model_2.obj = pyo.obj_2(rule=obj_2, sense=pyo.maximize)

    """constraints for subproblem"""
    #constraint for volume in t=25, the dual constraint
    def constraint_dual(model_2):
             return (model_2.V1_t[25] - model_2.I_2 + model_2.Q_t[25]== x_1) #legger til at output volum fra masterproblem er nå input volum i sub problem
    model_2.C6 = pyo.Constraint(rule=constraint_dual)

    # constraint 1, ensure that Q_t is lower than Qmax
    def constraint_Q(model_2, t):
        return model_2.Q_t[t] <= model_2.Q_Max
    model_2.C1 = pyo.Constraint(model_2.T_2, rule=constraint_Q)

     # constraint 2, constraint for water volume
    def constraint_V(model_2, t):
        if (t>=26):
            return (model_2.V1_t[t] == model_2.V1_t[t - 1] + model_2.I_2 - model_2.Q_t[t])
    model_2.C2 = pyo.Constraint(model_2.T_2, rule=constraint_V)

    # Constraint 3, ensure that V_t is lower than Vmax
    def constraint_V2(model_2, t):
        return (model_2.V1_t[t] <= model_2.V_MAX)
    model_2.C3 = pyo.Constraint(model_2.T_1, rule=constraint_V2)

    # Constraint 4, P_vt, kan legges i obj
    def constraint_P1(model_2, t):
         return (model_2.P_t[t] == model_2.Q_t[t] * model_2.E_conv * (1 / model_2.M_conv))
    model_2.C4 = pyo.Constraint(model_2.T_2, rule=constraint_P1)

    # Constraint 5, ensure that P_ts is lower than Pmax, trenger egentlig ikke
    def constraint_Pmax(model_2, t):
         return (model_2.P_ts[t] <= model_2.P_Max)
    model_2.C5 = pyo.Constraint(model_2.T_1, rule=constraint_Pmax)

    #løser solver her også også med dual
    solver = 'gurobi'
    opt = SolverFactory(solver, load_solution=True)
    results = opt.solve(model, load_solutions=True)
    model_2.display()
    model_2.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    dual = model_2.dual[model_2.C6]

    return model_2.OBJ2, model_2.dual


    def CreateCuts(OBJ2,dual,x_1):
        model_3 = pyo.ConcreteModel()

        model_3.OBJ2 = OBJ2
        model_3.dual = dual
        model_3.x_1 = x_1

        #hjelpevariabler:
        model_3.a = pyo.Var(model_3.a, domain=pyo.NonNegativeReals)  #
        model_3.b = pyo.Var(model_3.b, domain=pyo.NonNegativeReals)  #

        model_3.a = model_3.dual
        model_3.b = model_3.OBJ2 - model_3.dual*model_3.x_1

        return (model_3.a, model_3.b)
        #for s in range(S_range):
            #model_3.a = model_3.a + dual


