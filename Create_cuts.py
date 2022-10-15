import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from Master_Problem import MasterProblem


model.display()
Cuts_data = {} #lagrer informasjonen fra cutsene
List_of_cuts = [] #liste som er tom

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
# input()
