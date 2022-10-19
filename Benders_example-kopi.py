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
    model = pyo.ConcreteModel()

    """sets"""
    T_1_range = 24
    T_2_range = 48
    I_range = 20  # number of iterations
    S_range = 4  # scenarios
    model.T_1 = pyo.RangeSet(0, T_1_range)
    model.T_2 = pyo.RangeSet(25, T_2_range)
    model.S = pyo.RangeSet(0, S_range)
    model.Iteration = pyo.RangeSet(0, I_range)

    """parametere"""
    V_0 = 5  # starting volume in the reservoir given in Mm3 for t = 0
    V_MAX = 10  # maximum volume in the reservoir given in Mm3
    Q_Max = 0.36  # maximum outflow per hour from the reservoir given in Mm3/h
    P_MAX = 100  # maximum production per hour given in MW
    M_conv = 0.0036  # conversion factor given in [Mm3/m^3]
    E_conv = 0.981  # conversion factor for discharge water to produce electricity, [Mm3/m^3]
    WV_end = 13000  # end water value for all scenarios given in EUR/Mm3
    I_1 = 0.18  # Inflow during the first 24 hours
    I_2 = 0.18  # Inflow during the second 24 hours
    alpha = 0

    # initialize the parameters to the model:
    model.V_0 = pyo.Param(initialize=V_0)
    model.V_MAX = pyo.Param(initialize=V_MAX)
    model.Q_Max = pyo.Param(initialize=Q_Max)
    model.P_Max = pyo.Param(initialize=P_MAX)
    model.M_conv = pyo.Param(initialize=M_conv)
    model.E_conv = pyo.Param(initialize=E_conv)
    model.WV_end = pyo.Param(initialize=WV_end)
    # model.rho_s = pyo.Param(initialize=rho_s)
    model.I_1 = pyo.Param(initialize=I_1)
    model.I_2 = pyo.Param(initialize=I_2)
    model.alpha = pyo.Param(initialize=alpha)

    # price:
    Dict = {}
    for t in range(24):
        Dict[t] = 50 + t
    model.p_t = pyo.Param(model.T_1, initialize=Dict)
    print("Pris", Dict)

    Dict = {}
    for t in range(25, 48):
        Dict[t] = 50 + t
    model.p_t = pyo.Param(model.T_2, initialize=Dict)
    print("Pris", Dict)

    "Variables"
    model.P_t = pyo.Var(model.T_1, domain=pyo.NonNegativeReals)  # produced electricity
    model.Q_t = pyo.Var(model.T_1, domain=pyo.NonNegativeReals)  # outflow
    model.V1_t = pyo.Var(model.T_1, domain=pyo.NonNegativeReals)  # volume of water during the first 24 hours
    model.b = pyo.Var(model.Iteration, domain=pyo.NonNegativeReals)  # b i y = ax+b
    model.dual = pyo.Var(model.Iteration, domain=pyo.NonNegativeReals)  # a
    model.x_1 = pyo.Var(model.Iteration, domain=pyo.NonNegativeReals)

    # master
    Data= {} #lagrer alt av data vi vil ha
    Data[0]= "Master problem data:"
    print("data", Data)
    print("model", model)
    Data[1]= {} #Subproblems 
    for i in range(3): 
        Data[1][i]= i   #"Sub_problem_data_for_scenario_" + str(i)
        return(model)
    

def MasterProblem(model,iteration, Cuts_data):
    
    #This is the master problem variable output

    x_1 = iteration*2 
    print(model, x_1)
    print(Cuts_data)
    model.display()
    return(x_1) 
#in our problem we will also just return a variable 
#recreate it for each iteration 

def SubProblem(model, x_1):  #interested in the x_1 value
                                    # the sub problem will for each iteratiuon have uniqe data for each iteration 
                                    #for data which is stocastic or unsertant, make sure that you only give it the input that is for this specific scenario. 
   # model.param = pyo.Param(initialize = Data["Stochastic_Parameter"]) #only get dtaa in the spesific scenario (Data[1][i])
    
    print("SubProblem data:", model)
    OBJ = x_1*10*model
    Dual = (x_1/2)*model


    
   # model.param = pyo.Param(initialize = Data["Stochastic_Parameter"]) #only get dtaa in the spesific scenario (Data[1][i])
     
    return(OBJ, Dual, x_1)



def Create_cuts(model) :
    for iteration in range(10):

        """optimizatin problem"""
       # model = pyo.ConcreteModel()
        #model.x_l = pyo.Var()

        """constraints"""
        model.Cuts = pyo.Set(initialize=List_of_cuts) #tell us how many cust we have in the model
        model.Cuts_data = Cuts_data


        def Constraint_cuts(model, cut):
            print(model.Cuts_data[cut]["slope"], model.Cuts_data[cut]["constant"])
            print("Creating cut: ", cut)


            return (Cuts_data)
        model.Cut_constraint = pyo.Constraint(model.Cuts, rule=Constraint_cuts)

        # Create some cuts

        List_of_cuts.append(iteration)
        Cuts_data[iteration] = {}
        Cuts_data[iteration]["Slope"] = model.dual
        Cuts_data[iteration]["Constant"] = model.OBJ2 - model.dual*model.x_1
        model.display()
    return (Cuts_data)



x_1_data = {}
Cuts_data = {}
Preliminary_results = {} # save information regarding the cuts 

"""
Initiate the Benders Decomposition problem
"""

for iteration in range(1,10): 
    
    """
    Initiate the master problem 
    """
    
    #x_1 = MasterProblem(model, iteration, Cuts_data)

    x_1_data[iteration] = x_1
    
    """
    Initiate the sub problem 
    """
    Preliminary_results[iteration]= {}
    for scen in range(3):
        OBJ, Dual = SubProblem(model[1][scen], x_1)
        print("OBJ, Dual:", OBJ,Dual)
        Preliminary_results[iteration][scen] = {"OBJ":OBJ, "Dual": Dual, "x_1": x_1}
        
    """
    Create cuts
    """
    temp_data = Preliminary_results[iteration]
    Create_cuts(temp_data, Cuts_data)
    
    
    #sys.exit()

    