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

Cuts_data = {} #dictonary som skal inneholde alle cut
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
    #her lager vi masterproblemet
    model.x_1 = pyo.Var() #x_1 = alpha
    
    #Constraint for adding cuts, Dette er i Masterproblemet
    
    model.Cuts = pyo.Set(initialize = List_of_cuts) #set med indekser, første iterasjon vil det ikke være noe i lista
    model.Cuts.display()
    model.Cuts_data = Cuts_data #sier at den er lik en dictanary

    def Constraint_cuts(model,cut): # her vi må sette opp at alpha er <= osv
        print(model.Cuts_data[cut]["Slope"], model.Cuts_data[cut]["Constant"])
        print("Creating cut: ", cut)
        print("her printer jeg", Cuts_data)
        return(model.x_1 == 2)
    model.Cut_constraint = pyo.Constraint(model.Cuts, rule = Constraint_cuts) #denne constarinet vil genere 5 ganger siden vi har den inne i model:cuts
    
    #Create some cuts, denne delen
    List_of_cuts.append(iteration)  # added one cut for each iteration
    Cuts_data[iteration] = {}
    Cuts_data[iteration]["Slope"] = 30 * iteration
    Cuts_data[iteration]["Constant"] = 700 - iteration
    

    print("her printer jeg", Cuts_data)
    
input()
   