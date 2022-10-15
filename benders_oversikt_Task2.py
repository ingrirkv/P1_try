

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
model = pyo.ConcreteModel()

"""sets"""
T1_range = 24
T2_range = 48
I_range = 20 #number of iterations
S_range = 4 #scenarios
model.T1 = pyo.RangeSet(0,T1_range)
model.T2 = pyo.RangeSet(25,T2_range)
model.S = pyo.RangeSet(0,S_range)
model.Iteration = pyo.Rangeset(0,I_range)


"""parametere"""
V_0 = 5                #starting volume in the reservoir given in Mm3 for t = 0
V_MAX = 10             #maximum volume in the reservoir given in Mm3
Q_Max = 0.36           #maximum outflow per hour from the reservoir given in Mm3/h
P_MAX = 100            #maximum production per hour given in MW
M_conv = 0.0036        #conversion factor given in [Mm3/m^3]
E_conv = 0.981         #conversion factor for discharge water to produce electricity, [Mm3/m^3]
WV_end = 13000         #end water value for all scenarios given in EUR/Mm3
I_1 = 0.18             #Inflow during the first 24 hours
I_2 = 0.18             #Inflow during the second 24 hours
alpha = 0

#initialize the parameters to the model:
model.V_0 = pyo.Param(initialize=V_0)
model.V_MAX = pyo.Param(initialize=V_MAX)
model.Q_Max = pyo.Param(initialize=Q_Max)
model.P_Max = pyo.Param(initialize=P_MAX)
model.M_conv = pyo.Param(initialize=M_conv)
model.E_conv = pyo.Param(initialize=E_conv)
model.WV_end = pyo.Param(initialize=WV_end)
model.rho_s = pyo.Param(initialize=rho_s)
model.I_1 = pyo.Param(initialize=I_1)
model.I_2 = pyo.Param(initialize=I_2)
model.alpha= pyo.Param(initialize=alpha)

#price:
Dict={}
for t in range(49):
    Dict[t]=50+t
model.p_t = pyo.Param(model.T, initialize=Dict)
print("Pris", Dict)


"Variables"
model.P_t = pyo.Var(model.T1, domain=pyo.NonNegativeReals) #produced electricity
model.Q_t = pyo.Var(model.T1, domain=pyo.NonNegativeReals) #outflow
model.V1_t = pyo.Var(model.T1, domain=pyo.NonNegativeReals) #volume of water during the first 24 hours
model.b = pyo.Var(model.I, domain= pyo.NonNegativeReals) # b i y = ax+b
model.dual = pyo.Var(model.I, domain= pyo.NonNegativeReals) #a



"""
Initiate the Benders Decomposition problem
"""

for i in range(model.Iterations):

    """
    Initiate the master problem 
    """

    x_1 = MasterProblem(model[0], i, Cuts_data)

    x_1_data[i] = x_1

    """
    Initiate the sub problem 
    """
    Preliminary_results[i] = {}
    for scen in range(3):
        OBJ, Dual = SubProblem(Data[1][scen], x_1)
        print("OBJ, Dual:", OBJ, Dual)
        Preliminary_results[i][scen] = {"OBJ": OBJ, "Dual": Dual, "x_1": x_1}

    """
    Create cuts
    """
    temp_data = Preliminary_results[i]
    Create_cuts(temp_data, Cuts_data)

    # sys.exit()
