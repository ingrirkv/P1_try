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
T1_range = 24
T2_range = 48
I_range = 20
model.T1 = pyo.RangeSet(0,T1_range)
model.T2 = pyo.RangeSet(25,T2_range)
model.I = pyo.Rangeset(0,I_range)

for t in model.T:
    print(t)
model.display()


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
model.rho_s = pyo.Param(initialize=rho_s)
model.I_1 = pyo.Param(initialize=I_1)
model.I_2 = pyo.Param(initialize=I_2)
model.alpha= pyo.Param(initialize=alpha)

#price:
Dict={}
for t in range(49):
    Dict[t]=50+t
model.p_t = pyo.Param(model.T, initialize=Dict)
print("Pris", Dict)


"Variables"
model.P_t = pyo.Var(model.T1, domain=pyo.NonNegativeReals) #produced electricity
model.Q_t = pyo.Var(model.T1, domain=pyo.NonNegativeReals) #outflow
model.V1_t = pyo.Var(model.T1, domain=pyo.NonNegativeReals) #volume of water during the first 24 hours
#model.V2_t = pyo.Var(model.T, domain=pyo.NonNegativeReals) #voule of water during the second 24 hours
model.b = pyo.Var(model.I, domain= pyo.NonNegativeReals) # b i y = ax+b
model.dual = pyo.Var(model.I, domain= pyo.NonNegativeReals) #a


def Input_Data():
    Data = {}  # lagrer alt av data vi vil ha
    Data[0] = "Master problem data:"

    Data[1] = {}  # Data from Subproblems - ser bort fra dette per nå
    #for i in range(3):
        #Data[1][i] = i  # "Sub_problem_data_for_scenario_" + str(i)
        #return (Data)

#a = input_data()
cuts_data = {}
V_end_data = {}
Premiliminary_results = {}

#Dict_2={}
#for i in range(20): #vi har 10 iterations
  #  Dict_2[i]= #a*V_end + b
#model.alpha = pyo.Param(model.I, initialize=Dict)
#print("Pris", Dict)


"""MasterProblem"""
def MasterProblem(model, iteration, Cuts_data):
    """sets"""
    T1_range = 24
    I_range = 20
    model.T1 = pyo.RangeSet(0, T1_range)
    model.I = pyo.Rangeset(0, I_range)

    for t in model.T1:
        print(t)
    model.display()

    """parametere"""
    V_0 = 5  # starting volume in the reservoir given in Mm3 for t = 0
    V_MAX = 10  # maximum volume in the reservoir given in Mm3
    Q_Max = 0.36  # maximum outflow per hour from the reservoir given in Mm3/h
    P_MAX = 100  # maximum production per hour given in MW
    M_conv = 0.0036  # conversion factor given in [Mm3/m^3]
    E_conv = 0.981  # conversion factor for discharge water to produce electricity, [Mm3/m^3]
    WV_end = 13000  # end water value for all scenarios given in EUR/Mm3
    I_1 = 0.18  # Inflow during the first 24 hours
    #alpha = 0

    # initialize the parameters to the model:
    model.V_0 = pyo.Param(initialize=V_0)
    model.V_MAX = pyo.Param(initialize=V_MAX)
    model.Q_Max = pyo.Param(initialize=Q_Max)
    model.P_Max = pyo.Param(initialize=P_MAX)
    model.M_conv = pyo.Param(initialize=M_conv)
    model.E_conv = pyo.Param(initialize=E_conv)
    model.WV_end = pyo.Param(initialize=WV_end)
    model.I_1 = pyo.Param(initialize=I)
    

    # price:
    Dict = {}
    for t in range(25):
        Dict[t] = 50 + t
    model.p_t = pyo.Param(model.T, initialize=Dict)
    print("Pris", Dict)

    "Variables"
    model.P_t = pyo.Var(model.T1, domain=pyo.NonNegativeReals)  # produced electricity
    model.Q_t = pyo.Var(model.T1, domain=pyo.NonNegativeReals)  # outflow
    model.V1_t = pyo.Var(model.T1, domain=pyo.NonNegativeReals)  # volume of water during the first 24 hours
    model.b = pyo.Var(model.I, domain=pyo.NonNegativeReals)  # b i y = ax+b
    model.dual = pyo.Var(model.I, domain=pyo.NonNegativeReals)  # a
    model.alpha = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    # This is the master problem variable output
    
    
    """objective function"""
    OBJ1= sum(model.P_t[t] * model.p_t[t] for t in model.T1 + model.alpha) #Objective function


    """constraints for master problem"""
    # constraint 1, ensure that Q_t is lower than Qmax
    def constraint_Q(model, t):
        if t == 0:
            return model.Q_t[t] == 0
        else:
            return model.Q_t[t] <= model.Q_Max
    model.C1 = pyo.Constraint(model.T1, rule=constraint_Q)

    # constraint 2, constraint for water volume
    def constraint_V(model,t):
        if (t == 0):
            return (model.V1_t[0] == 5)
        else:
            return (model.V1_t[t] == model.V1_t[t - 1] + model.I_1 - model.Q_t[t])
    model.C2 = pyo.Constraint(model.T1, rule=constraint_V)

    # Constraint 3, ensure that V_t is lower than Vmax
    def constraint_V2(model,t):
        return (model.V1_t[t] <= model.V_MAX)
    model.C3 = pyo.Constraint(model.T1, rule=constraint_V2)

    # Constraint 4, P_vt, kan legges i obj
    def constraint_P1(model, t):
        return (model.P_t[t] == model.Q_t[t] * model.E_conv * (1 / model.M_conv))
    model.C4 = pyo.Constraint(model.T1, model.S, rule=constraint_P1)

    # Constraint 5, ensure that P_ts is lower than Pmax, trenger egentlig ikke
    def constraint_Pmax(model, t):
        return (model.P_ts[t] <= model.P_Max)
    model.C5 = pyo.Constraint(model.T1, rule=constraint_Pmax)

    #Constraint 6, alpha - denne må sjekkes
    #def constraint_alpha(model, i):
      #  return(model.alpha[i]<= model.dual*model.V_t+ b)
   # model.C5 = pyo.Constraint(model.I, rule=constraint_alpha)

    """return"""
    return (model.V1_t[24])
model.OBJ1 = pyo.Objective(rule=MasterProblem, sense=pyo.maximize)
model.display()




def SubProblem(Data, V1_t):
    """sets"""
    T2_range = 48
    I_range = 20
    model.T2 = pyo.RangeSet(25, T2_range)
    model.I = pyo.Rangeset(0, I_range)



    """parametere"""
    V_MAX = 10  # maximum volume in the reservoir given in Mm3
    Q_Max = 0.36  # maximum outflow per hour from the reservoir given in Mm3/h
    P_MAX = 100  # maximum production per hour given in MW
    M_conv = 0.0036  # conversion factor given in [Mm3/m^3]
    E_conv = 0.981  # conversion factor for discharge water to produce electricity, [Mm3/m^3]
    WV_end = 13000  # end water value for all scenarios given in EUR/Mm3
    I_2 = 0.18  # Inflow during the second 24 hour

    # initialize the parameters to the model:
    model.V_MAX = pyo.Param(initialize=V_MAX)
    model.Q_Max = pyo.Param(initialize=Q_Max)
    model.P_Max = pyo.Param(initialize=P_MAX)
    model.M_conv = pyo.Param(initialize=M_conv)
    model.E_conv = pyo.Param(initialize=E_conv)
    model.WV_end = pyo.Param(initialize=WV_end)
    model.I_2 = pyo.Param(initialize=I_2)

    # price:
    Dict = {}
    for t in range(25,49): #gå fra og med 25 til og med 48
        Dict[t] = 50 + t
    model.p_t = pyo.Param(model.T, initialize=Dict)
    print("Pris", Dict)

    "Variables"
    model.P_t = pyo.Var(model.T1, domain=pyo.NonNegativeReals)  # produced electricity
    model.Q_t = pyo.Var(model.T1, domain=pyo.NonNegativeReals)  # outflow
    model.V1_t = pyo.Var(model.T1, domain=pyo.NonNegativeReal)
    model.b = pyo.Var(model.I, domain=pyo.NonNegativeReals)  # b i y = ax+b
    model.dual = pyo.Var(model.I, domain=pyo.NonNegativeReals)  # a

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    
    # the sub problem will for each iteratiuon have uniqe data for each iteration
    # for data which is stocastic or unsertant, make sure that you only give it the input that is for this specific scenario.
    # model.param = pyo.Param(initialize = Data["Stochastic_Parameter"]) #only get dtaa in the spesific scenario (Data[1][i])

    """objective function"""
    OBJ2 = sum(model.P_t[t] * model.p_t[t] for t in model.T2) + model.WV_end*V1_t[24] #Objective function
    """dual"""
    #dual = #vi må finne ut hvordan vi regner dual

    """constraints for subproblem"""
    # constraint 1, ensure that Q_t is lower than Qmax
    def constraint_Q(model, t):
        return model.Q_t[t] <= model.Q_Max
    model.C1 = pyo.Constraint(model.T2, rule=constraint_Q)

    # constraint 2, constraint for water volume
    def constraint_V(model,t):
        if (t == 25):
            return (model.V1_t[t] == V1_t[24] + model.I_2 - model.Q_t[t])
        else:
            return (model.V1_t[t] == model.V1_t[t - 1] + model.I_2 - model.Q_t[t])
    model.C2 = pyo.Constraint(model.T2, rule=constraint_V)

    # Constraint 3, ensure that V_t is lower than Vmax
    def constraint_V2(model,t):
        return (model.V1_t[t] <= model.V_MAX)
    model.C3 = pyo.Constraint(model.T1, rule=constraint_V2)

    # Constraint 4, P_vt, kan legges i obj
    def constraint_P1(model, t):
        return (model.P_t[t] == model.Q_t[t] * model.E_conv * (1 / model.M_conv))
    model.C4 = pyo.Constraint(model.T1, model.S, rule=constraint_P1)

    # Constraint 5, ensure that P_ts is lower than Pmax, trenger egentlig ikke
    def constraint_Pmax(model, t):
        return (model.P_ts[t] <= model.P_Max)
    model.C5 = pyo.Constraint(model.T1, rule=constraint_Pmax)

    #constraint for dual

    print("SubProblem data:", V_t)

    """return"""
    return (Object, Dual)
model.OBJ2 = pyo.Objective(rule=SubProblem, sense=pyo.maximize)
    # model.param = pyo.Param(initialize = Data["Stochastic_Parameter"]) #only get dtaa in the spesific scenario (Data[1][i])



def Create_cuts(Data_raw, Cuts_data):
    Cuts_data = "I added cuts"
    return (Cuts_data)
