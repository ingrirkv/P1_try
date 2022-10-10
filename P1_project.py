#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:02:03 2022

@author: ingridrodahlkvale

"""

"HYDROPOWER PROBLEM "
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from __future__ import division


"Paramtere: "
"V_0 =5                 #[Mm3], million cubic meter, m3*10^6, The starting reservoir level for the hydropower station"
"V_MAX = 10              #[Mm3], Rated capacity for the reservoir"
"Q_Max = 100             #[m^3/s], Maximum discharge capacity of water from the reservoir to the hydropower unit"
"P_MAX = 100             #[MW], Maximum production capacity for the hydropower unit"
"M3S_TO_MM3= 3.6/1000    #[Mm3/m^3], Conversion factor from cubic meter persecond to Mm3 (million cubic meter)"
"E_conv = 0.981          #[MWh/m^3],Power equivalent from discharged water to produced electricity. "
"WV_end = 13000          #[EUR/Mm3], Water value for leftover hydropower at the end of the 48th hour."

"Indeks: "
"s= 0                    #scenario"
"t= 0                   #time step"


"Water value for leftover hydropower at the end of the 48th hour."

"Inflow_first_24h = 50   #[m^3/s], The inflow for the first 24 hours, given hourly"
"Inflow_Last_24h = 24*s  #[m^3/s], The inflow for the last 24 hours, given hourly. This includes uncertainty and a 0-index formulation. Example: For scenario 0, inflow is 10*0 = 0"
"N_Scenarios = 5         #[-], The number of scenarios for the second stage (last 24 hours)"
"ro_scenario = 0.2        #[per unit],  The probability for each scenario."
"Price = 50*t     "       #[EUR/MWh], The power prices for all 48 hours, given as a linearly increasing cost based on time step t. Assumes 0-index.Example: For hour 13 the cost is 50+13 = 63.

model = pyo.ConcreateModel()    '#Establish the optimization model, as a concrete model'

"model.m = pyo.Param(within = pyo.NonNegativeIntegers)"
"model.n = pyo.Param(within = pyo.NonNegativeIntegers)"

"sets"
model.s = pyo.RangeSet(0,4)
model.t = pyo.RangeSet(0,48)

"parametere"
V_0 =5                 #starting volume in the reservoir given in Mm3
V_MAX = 10             #maximum volume in the reservoir given in Mm3
Q_Max = 0.36           #maximum outflow per hour from the reservoir given in Mm3/h
P_MAX = 100            #maximum production per hour given in MW
M3S_TO_MM3 = 0.0036    #conversion factor
E_conv = 0.981         #conversion factor for discharge water to produce electricity
WV_end = 13000         #end water value for all scenarios given in EUR/Mm3
rho_s = 0.2            #probability for scenario s, equals 0.2 for all s
T = 48                 #number of hour
S = 5                  #number of scenarios

model.V_0 = pyo.Param(within = pyo.NonNegativeIntegers, initialize= 5)
model.rho_s = pyo.Param(model.s, initialize= 0.2)
model.Q_max = pyo.Param(model.t, model.s, initialize=Q_Max)
model.T = pyo.Param(initialize=48)


"Variables"
model.Z = pyo.Var()     #profitt
model.P_{t,s} = pyo.Var(model.t, model.s, domain = pyo.NonNegativeReals)                     #produsert elektrisitet [MWh]
model.I_{t,s} = pyo.Var(model.t,model.s, domain = pyo.NonNegativeReals, initialize = 25*s) #inflow
model.Q_{t,s} = pyo.Var(model.t,model.s, domain = pyo.NonNegativeReals)                     #outflow
model.p_{t} = pyo.Var(model.t, model.s, domain = pyo.NonNegativeReals, initialize=50*t) #pris
model.V_{t,s} = pyo.Var(model.t, model.s, domain = pyo.NonNegativeReals)

"model.OBJ = pyo.Objective(sum())"
def objective_func(m):
    return pyo.summation(m.P[t,s], m.p[t], m.ro_scenario[s])
model.OBJ = pyo.Objective(rule = objective_func)

def constraints():
    return()