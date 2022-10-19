#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 13:31:20 2022

@author: ingridrodahlkvale
"""
"""
Benders Decomposition
- Anbefales å lage funksjoner 
"""
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt


def Input_data():
    Data = {} #lagrer alt av data vi vil ha 
    Data[0]= "Master problem data:"
    
    Data[1]= {} #Subproblems 
    for i in range(3): 
        Data[1][i]= i   #"Sub_problem_data_for_scenario_" + str(i)
        return(Data)
    

def MasterProblem(Data,iteration, Cuts_data): 
    
    #This is the master problem variable output 
    x_1 = iteration*2 
    print(Data, x_1)
    print(Cuts_data)
    
    return(x_1) 
#in our problem we will also just return a variable 
#recreate it for each iteration 

def SubProblem(Data, x_1):  #interested in the x_1 value 
                                    # the sub problem will for each iteratiuon have uniqe data for each iteration 
                                    #for data which is stocastic or unsertant, make sure that you only give it the input that is for this specific scenario. 
   # model.param = pyo.Param(initialize = Data["Stochastic_Parameter"]) #only get dtaa in the spesific scenario (Data[1][i])
    
    print("SubProblem data:", Data)
    OBJ = x_1*10*Data
    Dual = (x_1/2)*Data
    
   # model.param = pyo.Param(initialize = Data["Stochastic_Parameter"]) #only get dtaa in the spesific scenario (Data[1][i])
     
    return(OBJ, Dual)
    
    
def Create_cuts(Data_raw, Cuts_data): 
    Cuts_data = "I added cuts"
    return(Cuts_data)
    
    
#this does not work, we have to figure it out 
    #for scen in Data_raw: 
        #Slope = Data_raw[scen]["Dual"]
        #Constant =  Data_raw[scen]["OBJ"]  #rember it is an example
   # try:
   # max_key = max(list(Cuts_data.keys()))
   # except: 
       # pass 
       # max_key = -1

Data = Input_data()

x_1_data = {}
Cuts_data = {}
Preliminary_results = {} # save information regarding the cuts 

"""
Initiate the Benders Decomposition problem
"""

for iteration in range(10):
    
    """
    Initiate the master problem 
    """
    
    x_1 = MasterProblem(Data[0], iteration, Cuts_data) # her kaller vi på masterfunksjonen

    x_1_data[iteration] = x_1
    
    """
    Initiate the sub problem 
    """
    Preliminary_results[iteration]= {}
    for scen in range(3):
        OBJ, Dual = SubProblem(Data[1][scen], x_1)
        print("OBJ, Dual:", OBJ,Dual)
        Preliminary_results[iteration][scen] = {"OBJ":OBJ, "Dual": Dual, "x_1": x_1}
        
    """
    Create cuts
    """
    temp_data = Preliminary_results[iteration]
    Create_cuts(temp_data, Cuts_data)
    
    
    #sys.exit()

    