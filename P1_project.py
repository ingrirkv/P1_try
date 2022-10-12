#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:02:03 2022

@author: ingridrodahlkvale

"""

"""HYDROPOWER PROBLEM """
#from __future__ import division
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt



#Establish the optimization model, as a concrete model
model = pyo.ConcreteModel()

"""sets"""
T_range = 48
S_range = 4

model.S = pyo.RangeSet(0,S_range)
model.T = pyo.RangeSet(0,T_range)

for t in model.T:
    print(t)
model.display()


"""parametere"""
V_0 = 5                #starting volume in the reservoir given in Mm3
V_MAX = 10             #maximum volume in the reservoir given in Mm3
Q_Max = 0.36           #maximum outflow per hour from the reservoir given in Mm3/h
P_MAX = 100            #maximum production per hour given in MW
M_conv = 0.0036        #conversion factor given in [Mm3/m^3]
E_conv = 0.981         #conversion factor for discharge water to produce electricity, [Mm3/m^3]
WV_end = 13000         #end water value for all scenarios given in EUR/Mm3
rho_s = 0.2            #probability for scenario s, equals 0.2 for all s

t = [i for i in range(50)]
print("dette er lista", t)

#initialize the parameters to the model:
model.V_0 = pyo.Param(initialize=V_0)
model.V_MAX = pyo.Param(initialize=V_MAX)
model.Q_Max = pyo.Param(initialize=Q_Max)
model.P_MAX = pyo.Param(initialize=P_MAX)
model.M_conv = pyo.Param(initialize=M_conv)
model.E_conv = pyo.Param(initialize=E_conv)
model.WV_end = pyo.Param(initialize=WV_end)
model.rho_s = pyo.Param(initialize=rho_s)

Dict={}
for t in range(49):
    Dict[t]= 50+t
model.p_t = pyo.Param(model.T, initialize=Dict)
print("Pris", Dict)


"Variables"
model.P_ts = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals) #produced electricity
model.Q_ts = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals) #outflow
model.V_ts = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals) #volume of water
model.I_ts = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals) #inflow , bruke et eksempel, kan hard kode


#forslag til objective function - er litt feil
#def objective_func(model):
    #for(t in model.T):
     #   for(s in model.S):

    #return pyo.summation(m.P_ts[t,s], m.p_ts[t], m.rho_s[s] for s in model.s for t in model.t) + pyo.summation(WV_end,V_ts[T,s] for s in model.s ):
#model.OBJ = pyo.Objective(rule = objective_func, sense = pyo.maximize)
#må vi dobbelsummere over t og s eller holder det å summere en gang slik som her?

def objective_rule (model):
    return sum(sum(model.P_ts[t,s]*model.p_t[t]*model.rho_s for s in model.S) for t in model.T) + sum(model.WV_end*model.V_ts[48,s] for s in model.S)
model.OBJ = pyo.Objective(rule = objective_rule, sense = pyo.maximize)



"""constraints"""
#constraint 1, ensure that Q_ts is lower than Qmax
def constraint_Q(model,t,s):
    return (model.Q_ts[t,s] <= model.Q_Max )
model.C1 = pyo.Constraint(model.T,model.S,rule=constraint_Q)

#constraint 2, constraint for water volume
def constraint_v1(model,t,s): #lage en hjelpevariabel
    if (t==0):
        return (model.V_ts[0,s] == 5)
    #return (model.V_ts[t, s] == model.V_ts[0, s] + 0 - model.Q_ts[t, s])
    else:
       # if s == 0  #alle scenarioer
        return (model.V_ts[t, s] == model.V_ts[t - 1, s] + model.I_ts[t, s] - model.Q_ts[t, s]) #endre I_ts til Inflow_ts
    model.C2 = pyo.Constraint(model.T, model.S, rule=constraint_v1)

#Constraint 3, ensure that V_ts is lower than Vmax
def constraint_V3 (model,t,s):
    return (model.V_ts[t,s] <= model.V_MAX)
model.C3 = pyo.Constraint(model.T, model.S, rule = constraint_V3)

#Constraint 4, P_vt
def constraint_P1(model,t,s):
    return (model.P_ts[t,s] == model.Q_ts[t,s]*model.E_conv*(1/model.M_conv))
model.C4 = pyo.Constraint(model.T, model.S, rule = constraint_P1)

#Constraint 5, ensure that P_ts is lower than Pmax
def constraint_Pmax(model,t,s):
    return (model.P_ts[t,s] <= model.P_Max)
model.C5 = pyo.Constraint(model.T, model.S, rule = constraint_Pmax)

#Constraint 6, price
#def constraint_pt(model,t): #trenger ikke om vi allerede har den som dictanary først
    #return (model.p_t[t]== 50+model.t)
#model.C6 = pyo.Constraint(model.T, rule = constraint_pt)
    
#Constraint 7, inflow
def constraint_I1(model,t,s):
    if (t==0):
         return (model.I_ts[t,s] ==0)
    elif (t <24):
        return (model.I_ts[t,s]== 0.18)
    else:
         return (model.I_ts[t, s] == 0.09*model.s)
model.C7 = pyo.Constraint(model.T, model.S, rule = constraint_I1 )


#Constraint 9, sikre at vi kun har ett utfall de første 24 timene
def constraint_P2(model):
    for t in range(24):
        return(model.P_ts[t,0]==model.P_ts[t,1]==model.P_ts[t,2]==model.P_ts[t,3]==model.P_ts[t,4])
model.C9 = pyo.Constraint(rule = constraint_P2)



"""Solve the problem"""

# Solve the problem,
solver = 'gurobi_ampl'
solver_io = 'nl'
opt = SolverFactory(solver ,solver_io=solver_io)
print("objective func: ", opt)

for s in model.S:
    print(model.s, model.t, model.V_ts, model.P_ts)

#opt = pyo.SolverFactory('glpk')
#results = opt.solve(model, stream-solver=true)

#endre