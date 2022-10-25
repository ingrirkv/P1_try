import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt



def MasterProblem(Cuts_data):  # ha med itaration,
    #This is the master problem variable output

    """set, variabler osv"""

    model = pyo.ConcreteModel()

    """sets"""
    T_1_range = 24
    model.T_1 = pyo.RangeSet(0, T_1_range)

    """parametere"""
    V_0 = 5  # starting volume in the reservoir given in Mm3 for t = 0
    V_MAX = 10  # maximum volume in the reservoir given in Mm3
    Q_Max = 0.36  # maximum outflow per hour from the reservoir given in Mm3/h
    P_MAX = 100  # maximum production per hour given in MW
    M_conv = 0.0036  # conversion factor given in [Mm3/m^3]
    E_conv = 0.981  # conversion factor for discharge water to produce electricity, [Mm3/m^3]
    I_1 = 0.18  # Inflow during the first 24 hours


    # initialize the parameters to the model:
    model.V_0 = pyo.Param(initialize=V_0)
    model.V_MAX = pyo.Param(initialize=V_MAX)
    model.Q_Max = pyo.Param(initialize=Q_Max)
    model.P_Max = pyo.Param(initialize=P_MAX)
    model.M_conv = pyo.Param(initialize=M_conv)
    model.E_conv = pyo.Param(initialize=E_conv)
    model.I_1 = pyo.Param(initialize=I_1)


    # price:
    Dict = {}
    for t in range(25):
        Dict[t] = 50 + t
    model.p_t = pyo.Param(model.T_1, initialize=Dict)
    #print("Pris", Dict)


    "Variables"
    model.P_t = pyo.Var(model.T_1, domain=pyo.NonNegativeReals)  # produced electricity
    model.Q_t = pyo.Var(model.T_1, domain=pyo.NonNegativeReals)  # outflow
    model.V1_t = pyo.Var(model.T_1, domain=pyo.NonNegativeReals)  # volume of water during the first 24 hours
    model.x_1 = pyo.Var(domain=pyo.NonNegativeReals)
    model.alpha = pyo.Var(domain=pyo.NonNegativeReals, bounds=(-10e10, 10e10))  # usikker på om alpha er en variabel

    """objective function"""
    def objective_func(model):
        obj_del1 = sum(model.P_t[t] * model.p_t[t] for t in model.T_1) + model.alpha  # Cuts_data.   # Objective function, er noe feil her for vi må egentlig bruke den Cuts_data
        return obj_del1
    model.obj_del1 = pyo.Objective(rule=objective_func, sense=pyo.maximize)


    """constraints for master problem"""
    # constraint 1, ensure that Q_t is lower than Qmax
    def constraint_Q(model, t):
        if t == 0:
            return model.Q_t[t] == 0
        else:
            return model.Q_t[t] <= model.Q_Max
    model.C1 = pyo.Constraint(model.T_1, rule=constraint_Q)

    # constraint 2, constraint for water volume
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
        return (model.P_t[t] == model.Q_t[t] * model.E_conv * (1/model.M_conv))
    model.C4 = pyo.Constraint(model.T_1, rule=constraint_P1)

    # Constraint 5, ensure that P_ts is lower than Pmax, trenger egentlig ikke
    def constraint_Pmax(model, t):
        return (model.P_t[t] <= model.P_Max)
    model.C5 = pyo.Constraint(model.T_1, rule=constraint_Pmax)

# en liste med cut
    List_of_cuts = []
    for el in Cuts_data:
        List_of_cuts.append(el)

    model.Cuts = pyo.Set(initialize=List_of_cuts)  # tell us how many cust we have in the model
    model.Cuts_data = Cuts_data # lager set med hvor mange cut vi har

    def Constraint_cuts(model, cut):
        print(model.Cuts_data[cut]["slope"], model.Cuts_data[cut]["constant"])
        print("Creating cut: ", cut)
        return(model.alpha <= model.Cuts_data[cut]["slope"] * model.V1_t[24] + model.Cuts_data[cut]["constant"])

    model.CC = pyo.Constraint(model.Cuts, rule=Constraint_cuts)


    # vi løser den her med solver osv
    solver = 'gurobi'
    opt = SolverFactory(solver, load_solution=True)
    results = opt.solve(model, load_solutions=True)
    #print("result", results)
    #model.display()
    print("max profit",pyo.value(model.obj_del1))
    x_1 = model.V1_t[24].value
    print("dette er verdien til x1:", x_1)
    return x_1


#subproblem
def SubProblem(x_1):
    model_2 = pyo.ConcreteModel()
    model_2.x_1 = x_1 #legger inn som parameter eller bare x_1

    """sets"""
    T_2_range = 48
    model_2.T_2 = pyo.RangeSet(25,T_2_range)

    """parametere"""
    V_MAX = 10             #maximum volume in the reservoir given in Mm3
    Q_Max = 0.36           #maximum outflow per hour from the reservoir given in Mm3/h
    P_MAX = 100            #maximum production per hour given in MW
    M_conv = 0.0036        #conversion factor given in [Mm3/m^3]
    E_conv = 0.981         #conversion factor for discharge water to produce electricity, [Mm3/m^3]
    WV_end = 13000         #end water value for all scenarios given in EUR/Mm3
    I_2 = 0.18             #Inflow during the second 24 hours


    #initialize the parameters to the model:

    model_2.V_MAX = pyo.Param(initialize=V_MAX)
    model_2.Q_Max = pyo.Param(initialize=Q_Max)
    model_2.P_Max = pyo.Param(initialize=P_MAX)
    model_2.M_conv = pyo.Param(initialize=M_conv)
    model_2.E_conv = pyo.Param(initialize=E_conv)
    model_2.WV_end = pyo.Param(initialize=WV_end)
    model_2.I_2 = pyo.Param(initialize=I_2)


    #price:
    # price:
    Dict_2 = {}
    for t in range(25,49):
        Dict_2[t] = 50 + t
    model_2.p_t_2 = pyo.Param(model_2.T_2, initialize=Dict_2)
    #print("Pris", Dict_2)

    "Variables"
    model_2.P_t = pyo.Var(model_2.T_2, domain=pyo.NonNegativeReals) #produced electricity
    model_2.Q_t = pyo.Var(model_2.T_2, domain=pyo.NonNegativeReals) #outflow
    model_2.V1_t = pyo.Var(model_2.T_2, domain=pyo.NonNegativeReals) #volume of water


#Subproblem
    def obj_2(model_2):
        OBJ2 = sum(model_2.P_t[t] * model_2.p_t_2[t] for t in model_2.T_2) + model_2.WV_end * model_2.V1_t[48]  # Objective function
        return OBJ2
    model_2.obj = pyo.Objective(rule=obj_2, sense=pyo.maximize)

    """constraints for subproblem"""
    #constraint for volume in t=25, the dual constraint
    def constraint_dual(model_2):
             return (model_2.V1_t[25] - model_2.I_2 + model_2.Q_t[25]== model_2.x_1) #legger til at output volum fra masterproblem er nå input volum i sub problem
    model_2.C6 = pyo.Constraint(rule=constraint_dual)

    def constraint_V3(model_2):
        return (model_2.V1_t[25] == model_2.SV + model_2.I_2 - model_2.Q_t[25]) #sjekk denne mer
    model_2.C7 = pyo.Constraint(rule=constraint_V3)

    # constraint 1, ensure that Q_t is lower than Qmax
    def constraint_Q(model_2, t):
        return model_2.Q_t[t] <= model_2.Q_Max
    model_2.C1 = pyo.Constraint(model_2.T_2, rule=constraint_Q)

     # constraint 2, constraint for water volume
    def constraint_V(model_2, t):
        if (t>=26):
            return (model_2.V1_t[t] == model_2.V1_t[t - 1] + model_2.I_2 - model_2.Q_t[t])
        else:
            return pyo.Constraint.Skip
    model_2.C2 = pyo.Constraint(model_2.T_2, rule=constraint_V)

    # Constraint 3, ensure that V_t is lower than Vmax
    def constraint_V2(model_2, t):
        return (model_2.V1_t[t] <= model_2.V_MAX)
    model_2.C3 = pyo.Constraint(model_2.T_2, rule=constraint_V2)

    # Constraint 4, P_vt, kan legges i obj
    def constraint_P1(model_2, t):
         return (model_2.P_t[t] == model_2.Q_t[t] * model_2.E_conv * (1 / model_2.M_conv))
    model_2.C4 = pyo.Constraint(model_2.T_2, rule=constraint_P1)

    # Constraint 5, ensure that P_ts is lower than Pmax, trenger egentlig ikke
    def constraint_Pmax(model_2, t):
         return (model_2.P_t[t] <= model_2.P_Max)
    model_2.C5 = pyo.Constraint(model_2.T_2, rule=constraint_Pmax)

    #løser solver her også også med dual
    model_2.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    solver = 'gurobi'
    opt = SolverFactory(solver, load_solution=True)
    results = opt.solve(model_2, load_solutions=True)
    #model_2.display()
    dual = model_2.dual[model_2.C6]
    obj_2 = pyo.value(model_2.obj)
    print("dual er :",dual)


    return obj_2, dual

def CreateCuts(obj_2, dual, x_1):

    a = dual
    b = obj_2 - dual *x_1
    return (a,b)



Cuts_data = {}
#må ikke vi fortelle Cuts_Data at første verdi er cut nr, så er det "slope" så "constant"

for it in range (10):

    x_1 = MasterProblem(Cuts_data)
    print("x_1 har verdi:", x_1)

    #for s in range(4):
    obj_2, dual = SubProblem(x_1)

    print("obj_2:", obj_2," dual", dual)
    a, b = CreateCuts(obj_2,dual,x_1)
    print("it:", it, "a: ", a, "b:",b)

    #List_of_cuts.append(it)
    #Cuts_data.append(it)
    Cuts_data[it] = {}
    Cuts_data[it]["slope"] = a
    Cuts_data[it]["constant"] = b



