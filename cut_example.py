#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 11:28:44 2022

@author: ingridrodahlkvale
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib

Cuts_data = {}
#Cuts_data[0] = {"Slope": 30, "Constant":700}

List_of_cuts = []
'''
key - Number for specific cut
Constant- Constant value for linear function
Slope - Slope value for the linear function
'''


for iteration in range(10):
    
    #Create an optimization problem
    
    model = pyo.ConcreteModel()
    model.x_l = pyo.Var() #må gjøre noe
    
    #Constraint for adding cuts
    
    model.Cuts = pyo.Set(initialize = List_of_cuts)
    model.Cuts_data = Cuts_data

    def Constraint_cuts(model,cut):
        print(model.Cuts_data[cut]["Slope"], model.Cuts_data[cut]["Constant"])
        print("Creating cut: ", cut)



        print("her printer jeg", Cuts_data)

        return(model.x_l == 2)

    model.Cut_constraint = pyo.Constraint(model.Cuts, rule = Constraint_cuts)
    
    #Create some cuts
    List_of_cuts.append(iteration)  # added one cut for each iteration
    Cuts_data[iteration] = {}
    Cuts_data[iteration]["Slope"] = 30 * iteration
    Cuts_data[iteration]["Constant"] = 700 - iteration
    

    print("her printer jeg", Cuts_data)
    
input()
   