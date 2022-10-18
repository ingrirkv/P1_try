import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
model = pyo.ConcreteModel()

"""sets"""
T_1_range = 24
T_2_range = 48
I_range = 20 #number of iterations
S_range = 4 #scenarios
model.T_1 = pyo.RangeSet(0,T_1_range)
model.T_2 = pyo.RangeSet(0,T_2_range)
model.S = pyo.RangeSet(0,S_range)
model.Iteration = pyo.RangeSet(0,I_range)


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
#model.rho_s = pyo.Param(initialize=rho_s)
model.I_1 = pyo.Param(initialize=I_1)
model.I_2 = pyo.Param(initialize=I_2)
model.alpha= pyo.Param(initialize=alpha)

#price:
Dict={}
for t in range(24):
    Dict[t]=50+t
model.p_t = pyo.Param(model.T_1, initialize=Dict)
print("Pris", Dict)

Dict={}
for g in range(25, 48):
    Dict[g]= 50+g
model.p_t = pyo.Param(model.T_2, initialize=Dict)
print("Pris", Dict)

"Variables"
model.P_t = pyo.Var(model.T_1, domain=pyo.NonNegativeReals) #produced electricity
model.Q_t = pyo.Var(model.T_1, domain=pyo.NonNegativeReals) #outflow
model.V1_t = pyo.Var(model.T_1, domain=pyo.NonNegativeReals) #volume of water during the first 24 hours
model.b = pyo.Var(model.Iteration, domain= pyo.NonNegativeReals) # b i y = ax+b
model.dual = pyo.Var(model.Iteration, domain= pyo.NonNegativeReals) #a
model.x_1 = pyo.Var(model.Iteration, domain= pyo.NonNegativeReals)

#master


def MasterProblem(model, Cuts_data): #ha med itaration,
    # This is the master problem variable output
    """objective function"""
    OBJ1 = sum(model.P_t[t] * model.p_t[t] for t in model.T_1 )  # Objective function, er noe feil her for vi må egentlig bruke den Cuts_data
    x_1 = pyo.value(model.V1_t[24])
    print(OBJ1)
    print("x_1:", x_1)
    return (x_1)
model.OBJ1 = pyo.Objective(rule = MasterProblem, sense = pyo.maximize)
    #"""constraints for master problem"""
    # constraint 1, ensure that Q_t is lower than Qmax
def constraint_Q(model, t):
        if t == 0:
            return model.Q_t[t] == 0
        else:
            return model.Q_t[t] <= model.Q_Max
model.C1 = pyo.Constraint(model.T_1, rule=constraint_Q)

#constraint 2, constraint for water volume
def constraint_V(model, t):
    if (t == 0):
        return (model.V1_t[0] == 5)
    else:
        return (model.V1_t[t] == model.V1_t[t - 1] + model.I_1 - model.Q_t[t])
model.C2 = pyo.Constraint(model.T_1, rule=constraint_V)

# Constraint 3, ensure that V_t is lower than Vmax
def constraint_V2(model, t):
    return (model.V1_t[t] <= model.V_MAX)
model.C3 = pyo.Constraint(model.T_1, rule=constraint_V2)

# Constraint 4, P_vt, kan legges i obj
def constraint_P1(model, t):
    return (model.P_t[t] == model.Q_t[t] * model.E_conv * (1 / model.M_conv))
model.C4 = pyo.Constraint(model.T_1, rule=constraint_P1)

# Constraint 5, ensure that P_ts is lower than Pmax, trenger egentlig ikke
def constraint_Pmax(model, t):
    return (model.P_ts[t] <= model.P_Max)
model.C5 = pyo.Constraint(model.T_1, rule=constraint_Pmax)

# Constraint 6, alpha - må sikre at denne kun leses for iteration 1, 2 ,3 ,4 osv. og ikke for iteration 0
def constraint_alpha(model, Cuts_data):
    if(model.OBJ2 >=model.alpha <= Cuts_data[i-1]("Slope")*model.V1_t[24] + Cuts_data[i-1]("Constant") for i in model.Iteration):
            return (model.alpha == model.OBJ2) #forstår koden at model.OBJ2 er profitten i forrige subproblem
    else:
        return(model.alpha == Cuts_data[i-1]("Slope")*model.V1_t[24] + Cuts_data[i-1]("Constant") for i in model.Iteration)
model.C5 = pyo.Constraint(model.Iteration, model.aplha, rule=constraint_alpha)

#model.OBJ1 = pyo.Objective(rule=MasterProblem, sense=pyo.maximize)


#SubProblem
def SubProblem(model,x_1): #må finne ut
    """objective function"""
    OBJ2 = sum(model.P_t[t] * model.p_t[t] for t in model.T2) + model.WV_end * model.V1_t[48]  # Objective function
    return OBJ2, Dual
model.OBJ2 = pyo.Objective(rule=SubProblem, sense=pyo.maximize)

    #"""constraints for subproblem"""
    #constraint for volume in t=25, the dual constraint
def constraint_dual(model,t):
    if (t==25):
         return (model.V1_t[24] == model.x_1) #legger til at output volum fra masterproblem er nå input volum i sub problem
    else:
        return pyo.Constraint.Skip
model.C6 = pyo.Constraint(model.T_2, rule=constraint_dual)

    # constraint 1, ensure that Q_t is lower than Qmax
def constraint_Q(model, t):
    return model.Q_t[t] <= model.Q_Max
model.C1 = pyo.Constraint(model.T_2, rule=constraint_Q)

    # constraint 2, constraint for water volume
def constraint_V(model, t):
    if (t>=26):
        return (model.V1_t[t] == model.V1_t[t - 1] + model.I_2 - model.Q_t[t])
model.C2 = pyo.Constraint(model.T_2, rule=constraint_V)

# Constraint 3, ensure that V_t is lower than Vmax
def constraint_V2(model, t):
    return (model.V1_t[t] <= model.V_MAX)
model.C3 = pyo.Constraint(model.T_1, rule=constraint_V2)

# Constraint 4, P_vt, kan legges i obj
def constraint_P1(model, t):
     return (model.P_t[t] == model.Q_t[t] * model.E_conv * (1 / model.M_conv))
model.C4 = pyo.Constraint(model.T_1, model.S, rule=constraint_P1)

# Constraint 5, ensure that P_ts is lower than Pmax, trenger egentlig ikke
def constraint_Pmax(model, t):
     return (model.P_ts[t] <= model.P_Max)
model.C5 = pyo.Constraint(model.T_1, rule=constraint_Pmax)


    #dual
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
Dual = model.dual(model.constraint_dual[1])
model.display()
print("Resvervoir level:", model.x_1)
print("Objective function form Sub", model.OBJ2)
print ("Dual value", Dual)

 #skal disse ligge i subproblem eller create cuts
#Cuts_data[i]("slope") = model.Dual
#Cuts_data[i]("constant") = model.OBJ2 - model.Dual *x_1

#Crete cuts
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

def Create_cuts(Cuts_data):
    return(alpha)
model.Create_cuts= pyo.Constraint(model.Cuts, rule = Create_cuts)

Data = model

X_1_data = {}
Cuts_data = {}
Preliminary_results = {}

"""
Initiate the Benders Decomposition problem
"""

for i in range(model.Iterations):

    """
    Initiate the master problem 
    """

    x_1 = MasterProblem(model[0], i, Cuts_data)
    print("Dette er en test!", x_1)
    X_1_data[i] = x_1


    """
    Initiate the sub problem 
    """
    Preliminary_results[i] = {}
    OBJ, Dual = SubProblem(Data[1], x_1)
    print("OBJ, Dual:", OBJ, Dual)
    Preliminary_results[i] = {"OBJ": OBJ, "Dual": Dual, "x_1": x_1}

    """
    Create cuts
    """
    temp_data = (model.OBJ2, model.Dual)
    Create_cuts(temp_data, Cuts_data)

#skriver ut modellen
model.display()
model.dual.display()