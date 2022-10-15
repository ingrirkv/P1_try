

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import Sub_problem as Sub_problem
import Create_cuts as Create_cut

model = pyo.ConcreteModel() #må man ha den hver gang ?

def MasterProblem(model, iteration, Cuts_data):
    # This is the master problem variable output

    print(model.V_t)
    print(Cuts_data)

    """objective function"""
    OBJ1 = sum(model.P_t[t] * model.p_t[t] for t in model.T1 + model.alpha)  # Objective function
    x_1 = model.V1_t[24]
    return (x_1)
model.OBJ1 = pyo.Objective(rule=MasterProblem, sense=pyo.maximize)
    #"""constraints for master problem"""
    # constraint 1, ensure that Q_t is lower than Qmax
def constraint_Q(model, t):
        if t == 0:
            return model.Q_t[t] == 0
        else:
            return model.Q_t[t] <= model.Q_Max
model.C1 = pyo.Constraint(model.T1, rule=constraint_Q)

#constraint 2, constraint for water volume
def constraint_V(model, t):
    if (t == 0):
        return (model.V1_t[0] == 5)
    else:
        return (model.V1_t[t] == model.V1_t[t - 1] + model.I_1 - model.Q_t[t])
model.C2 = pyo.Constraint(model.T1, rule=constraint_V)

# Constraint 3, ensure that V_t is lower than Vmax
def constraint_V2(model, t):
    return (model.V1_t[t] <= model.V_MAX)
model.C3 = pyo.Constraint(model.T1, rule=constraint_V2)

# Constraint 4, P_vt, kan legges i obj
def constraint_P1(model, t):
    return (model.P_t[t] == model.Q_t[t] * model.E_conv * (1 / model.M_conv))
model.C4 = pyo.Constraint(model.T1, rule=constraint_P1)

# Constraint 5, ensure that P_ts is lower than Pmax, trenger egentlig ikke
def constraint_Pmax(model, t):
    return (model.P_ts[t] <= model.P_Max)
model.C5 = pyo.Constraint(model.T1, rule=constraint_Pmax)

# Constraint 6, alpha - må sikre at denne kun leses for iteration 1, 2 ,3 ,4 osv. og ikke for iteration 0
def constraint_alpha(model, Cuts_data):
    if(model.OBJ2 >=model.alpha <= Cuts_data[i-1]("Slope")*model.V1_t[24] + Cuts_data[i-1]("Constant") for i in model.Iteration):
            return (model.alpha == model.OBJ2) #forstår koden at model.OBJ2 er profitten i forrige subproblem
    else:
        return(model.alpha == Cuts_data[i-1]("Slope")*model.V1_t[24] + Cuts_data[i-1]("Constant") for i in model.Iteration)
model.C5 = pyo.Constraint(model.Iteration, model.aplha, rule=constraint_alpha)


"""return"""


#model.OBJ1 = pyo.Objective(rule=MasterProblem, sense=pyo.maximize)
model.display()







