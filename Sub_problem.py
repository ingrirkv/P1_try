import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

#Establish the optimization model, as a concrete model
model = pyo.ConcreteModel()


def SubProblem(model, x_1):
    """objective function"""
    OBJ2 = sum(model.P_t[t] * model.p_t[t] for t in model.T2) + model.WV_end * model.V1_t[48]  # Objective function
    return OBJ2, Dual
model.OBJ2 = pyo.Objective(rule=SubProblem, sense=pyo.maximize)

    """constraints for subproblem"""
    #constraint for volume in t=25, the dual constraint
def constraint_dual(model,t):
    if (t==25):
         return (model.V1_t[24] == x_1) #legger til at output volum fra masterproblem er nå input volum i sub problem
    else:
        return pyo.Constraint.Skip
model.C6 = pyo.Constraint(model.T2, rule=constraint_dual)

    # constraint 1, ensure that Q_t is lower than Qmax
def constraint_Q(model, t):
    return model.Q_t[t] <= model.Q_Max

model.C1 = pyo.Constraint(model.T2, rule=constraint_Q)

    # constraint 2, constraint for water volume
def constraint_V(model, t):
    if (t>=26):
        return (model.V1_t[t] == model.V1_t[t - 1] + model.I_2 - model.Q_t[t])
    else:
        return pyo.Constraint.Skip
 model.C2 = pyo.Constraint(model.T2, rule=constraint_V)

# Constraint 3, ensure that V_t is lower than Vmax
def constraint_V2(model, t):
    return (model.V1_t[t] <= model.V_MAX)
model.C3 = pyo.Constraint(model.T1, rule=constraint_V2)

# Constraint 4, P_vt, kan legges i obj
def constraint_P1(model, t):
     return (model.P_t[t] == model.Q_t[t] * model.E_conv * (1 / model.M_conv))
model.C4 = pyo.Constraint(model.T1, model.S, rule=constraint_P1)

# Constraint 5, ensure that P_ts is lower than Pmax, trenger egentlig ikke
def constraint_Pmax(model, t):
     return (model.P_ts[t] <= model.P_Max)
model.C5 = pyo.Constraint(model.T1, rule=constraint_Pmax)


    #dual
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
Dual = model.dual(model.constraint_dual[1])
model.display()
print("Resvervoir level:", model.x_1)
print("Objective function form Sub", model.OBJ2)
print ("Dual value", Dual)

 #skal disse ligge i subproblem eller create cuts
Cuts_data[i]("slope") = model.Dual
Cuts_data[i]("constant") = model.OBJ2 - model.Dual *x_1

# model.param = pyo.Param(initialize = Data["Stochastic_Parameter"]) #only get dtaa in the spesific scenario (Data[1][i])
