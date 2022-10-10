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

"Paramtere: "
    
V_0 = 5 ;               #[Mm3], million cubic meter, m3*10^6, The starting reservoir level for the hydropower station
V_MAX = 10 ;            #[Mm3], Rated capacity for the reservoir
Q_Max = 100             #[m^3/s], Maximum discharge capacity of water from the reservoir to the hydropower unit
P_MAX = 100             #[MW], Maximum production capacity for the hydropower unit
M3S_TO_MM3= 3.6/1000    #[Mm3/m^3], Conversion factor from cubic meter persecond to Mm3 (million cubic meter)
E_conv = 0.981          #[MWh/m^3],Power equivalent from discharged water to produced electricity.  
WV_end = 13000          #[EUR/Mm3], Water value for leftover hydropower at the end of the 48th hour.


"Variables: "
s= 0 
t= 0

"Water value for leftover hydropower at the end of the 48th hour."

Inflow_first_24h = 50   #[m^3/s], The inflow for the first 24 hours, given hourly
Inflow_Last_24h = 24*s  #[m^3/s], The inflow for the last 24 hours, given hourly. This includes uncertainty and a 0-index formulation. Example: For scenario 0, inflow is 10*0 = 0
N_Scenarios = 5         #[-], The number of scenarios for the second stage (last 24 hours)
p_scenario = 1/5        #[per unit],  The probability for each scenario.
Price = 50*t            #[EUR/MWh], The power prices for all 48 hours, given as a linearly increasing cost based on time step t. Assumes 0-index.Example: For hour 13 the cost is 50+13 = 63.


"dette er en test 2"