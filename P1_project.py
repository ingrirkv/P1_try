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




#Inflow_first_24h = 50   #[m^3/s], The inflow for the first 24 hours, given hourly"
#Inflow_Last_24h = 25*s  #[m^3/s], The inflow for the last 24 hours, given hourly. This includes uncertainty and a 0-index formulation. Example: For scenario 0, inflow is 10*0 = 0"

#Establish the optimization model, as a concrete model
model = pyo.ConcreteModel()
"""sets"""
model.T_range = pyo.Param(initialize=48)
model.S_range = pyo.Param(initialize=4)

model.S = pyo.RangeSet(model.S_range)
model.T = pyo.RangeSet(model.T_range)


"""parametere"""
V_0 = 5                #starting volume in the reservoir given in Mm3
V_MAX = 10             #maximum volume in the reservoir given in Mm3
Q_Max = 0.36           #maximum outflow per hour from the reservoir given in Mm3/h
P_MAX = 100            #maximum production per hour given in MW
M_conv = 0.0036        #conversion factor given in [Mm3/m^3]
E_conv = 0.981         #conversion factor for discharge water to produce electricity, [Mm3/m^3]
WV_end = 13000         #end water value for all scenarios given in EUR/Mm3
rho_s = 0.2            #probability for scenario s, equals 0.2 for all s



#initialize the parameters to the model:
model.V_0 = pyo.Param(initialize=V_0)
model.V_MAX = pyo.Param(initialize=V_MAX)
model.Q_max = pyo.Param(initialize=Q_Max)
model.P_MAX = pyo.Param(initialize=P_MAX)
model.M_conv = pyo.Param(initialize=M_conv)
model.E_conv = pyo.Param(initialize=E_conv)
model.WV_end = pyo.Param(initialize=WV_end)
model.rho_s = pyo.Param(initialize=rho_s)

model.I_ts = pyo.Param(model.T, model.S, initialize=0) #inflow
model.p_t = pyo.Param(model.T, initialize=50) #price

#vi definerer alle variabler som tomme uten verdi, slik de er forklart i latex filen. er dette da gjort riktig?
# eller skal alle parantesene stå tomme?
#hva er forskjell på within og domain?
"Variables"
#model.z = pyo.Var(within = pyo.NonNegativeReals)     #profitt
model.P_ts = pyo.Var(model.T, model.S, domain = pyo.NonNegativeReals) #produced electricity#, initialize = E_conv*M3S_TO_MM3)                     #produsert elektrisitet [MWh]
#model.I_ts = pyo.Var(model.T,model.S, domain = pyo.NonNegativeReals) #inflow #, initialize = 25*model.s) #inflow
model.Q_ts = pyo.Var(model.T,model.S, domain = pyo.NonNegativeReals) #outflow              #outflow
#model.p_t = pyo.Var(model.t, domain = pyo.NonNegativeReals) #price # initialize=50*model.t) #pris
model.V_ts = pyo.Var(model.T, model.S, domain = pyo.NonNegativeReals) #volume of water

"Functions"
#def price(model):
#    pris = (50+model.t)
#    return pris  #vil skrive ut funksjon 50+t = p_ts, og gjenbruke senere i koden


def Inflow(m,t,s):
    if value(model.t) = 0:
        return (model.I_ts = 0)
    elif value(model.t) > 0 && value(model.t) <= 24:
        return (model.I_ts = 0.18)
    else
        return (model.I_ts = 0.09*model.s)

def Production(m,t,s): #må kanskje ta inn Q_ts? tror kanskje ikke denne skal defineres, men legges rett som constraint
    return(Q_ts * E_conv * 1/M3S_TO_MM3)

"model.OBJ = pyo.Objective(sum())"
def objective_func(model):
    for(t in model.T):
        for(s in model.S):

    return pyo.summation(m.P_ts[t,s], m.p_ts[t], m.rho_s[s] for s in model.s for t in model.t) + pyo.summation(WV_end,V_ts[T,s] for s in model.s ):
model.OBJ = pyo.Objective(rule = objective_func, sense = pyo.maximize)
#må vi dobbelsummere over t og s eller holder det å summere en gang slik som her?

def objective_rule (model):
    return sum(sum(model.P_ts[t,s]*model.p_t[t]*model.rho_s for s in model.S) for t in model.T) + sum(model.WV_end*model.V_ts[48,s] for s in model.S)
model.OBJ = pyo.Objective(rule = objective_rule, sense = pyo.maximize)


def constraint_():

    #const1 0 <= Q_ts <= Q_Max
    #const2 V_ts = V_(t-1)s + I_ts - Q_ts
    #const3 V_ts <= V_MAX
    #const4 Production = Pts = (Qts*Econv)/Mconv
    #const5 price = 50 +t
    #const6 Inflow = 0.09*s
    #const7 non-negativity constraint
    return()





    def Constraint_assemble(model):
        return (3 * model.Alu + 2 * model.Wood <= 18)

    model.C1 = pyo.Constraint(rule=Constraint_assemble)

    # Pretty much the same setup as for creating the objective function

    def Constraint_AMan(model):
        return (1 * model.Alu <= 4)

    model.C2 = pyo.Constraint(rule=Constraint_AMan)

    def Constraint_WMan(model):
        return (2 * model.Wood <= 12)

    model.C3 = pyo.Constraint(rule=Constraint_WMan)

    #ida prøver å lage constraints

    #constraint 1, skal gjelde for alle t og alle s
    def constraint_Q(model,t,s):
        return (model.Q_ts[t,s] <= model.Q_Max )
    model.C1 = pyo.Constraint(rule=constraint_Q)

    #constraint 2, her må vi sikre at dette kun gjelder for t=0
    def constraint_v1(model,t,s):
        if (t==0):
            return (model.V_ts[0,s] == 5)
        else:
            return (model.V_ts[t, s] == model.V_ts[t - 1, s] + model.I_ts[t, s] - model.Q_ts[t, s])
    model.C2 = pyo.Constraint(rule = constraint_v1)


    #Constraint 4  : må man sette inn modle.s og model.t når det e rvariabel i constraint?
    def constraint_V3 (model,t,s):
        return (model.V_ts[t,s] <= model.V_MAX)
    model.C4 = pyo.Constraint(rule = constraint_V3)

    #Constraint 5:
    def constraint_Pts(model,t,s):
        return (model.P_ts[t,s] == model.Q_ts[t,s]*model.E_conv*(1/model.M_conv))
    model.C5 = pyo.Constraint(rule = constraint_Pts)
        

    #Constraint 6:
    def constraint_pt(model,t):
        return (model.p_t[t]== 50+model.t)
    model.C6 = pyo.Constraint(rule = constraint_pt)
    
#Constraint 7, inflow
def constraint_I1(model,t,s):
    if (t==0):
         return model.I_ts[t,s] ==0
    elif (t <24):
        return (model.I_ts[t,s]== 0.18)
    else:
         return (model.I_ts[t, s] == 0.09*model.s)
model.C7 = pyo.Constraint(rule = constraint_I1 )



"""Solve the problem"""

# Solve the problem
opt = pyo.SolverFactory()
result = opt.solve(model, load_solutions=True)
print("objective func: ", result)



#opt = pyo.SolverFactory('glpk')
#results = opt.solve(model, stream-solver=true)