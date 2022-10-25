
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt


#subproblem
def SubProblem(n):
    model_2 = pyo.ConcreteModel()
    model_2.x_1 = n #legger inn som parameter eller bare x_1

    """sets"""
    T_2_range = 48
    S_2_range = 4
    model_2.T_2 = pyo.RangeSet(25,T_2_range)
    model_2.S = pyo.RangeSet(0,S_2_range)

    """parametere"""
    V_MAX = 10             #maximum volume in the reservoir given in Mm3
    Q_Max = 0.36           #maximum outflow per hour from the reservoir given in Mm3/h
    P_MAX = 100            #maximum production per hour given in MW
    M_conv = 0.0036        #conversion factor given in [Mm3/m^3]
    E_conv = 0.981         #conversion factor for discharge water to produce electricity, [Mm3/m^3]
    WV_end = 13000         #end water value for all scenarios given in EUR/Mm3
    I_2 = 0.09             #Inflowconstant
    rho = 0.2              #sannsynlighet for hvert scenario


    #initialize the parameters to the model:

    model_2.V_MAX = pyo.Param(initialize=V_MAX)
    model_2.Q_Max = pyo.Param(initialize=Q_Max)
    model_2.P_Max = pyo.Param(initialize=P_MAX)
    model_2.M_conv = pyo.Param(initialize=M_conv)
    model_2.E_conv = pyo.Param(initialize=E_conv)
    model_2.WV_end = pyo.Param(initialize=WV_end)
    model_2.I_2 = pyo.Param(initialize=I_2)
    model_2.rho = pyo.Param(initialize=rho)

    #price:
    # price:
    Dict_2 = {}
    for t in range(25,49):
        Dict_2[t] = 50 + t
    model_2.p_t_2 = pyo.Param(model_2.T_2, initialize=Dict_2)
    #print("Pris", Dict_2)

    "Variables"
    model_2.P_ts = pyo.Var(model_2.T_2,model_2.S, domain=pyo.NonNegativeReals) #produced electricity
    model_2.Q_ts = pyo.Var(model_2.T_2,model_2.S, domain=pyo.NonNegativeReals) #outflow
    model_2.V1_ts = pyo.Var(model_2.T_2,model_2.S, domain=pyo.NonNegativeReals) #volume of water
    model_2.SV = pyo.Var(domain=pyo.NonNegativeReals)

#Subproblem
    def objective_rule(model_2):  # husk  legge til rho i siste sum
        OBJ2 = sum(sum(model_2.P_ts[t, s] * model_2.p_t_2[t] * model_2.rho for s in model_2.S) for t in model_2.T_2)+sum(model_2.WV_end * model_2.V1_ts[48, s] * model_2.rho for s in model_2.S)
        return OBJ2
    model_2.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    """constraints for subproblem"""
    #constraint for volume in t=25, the dual constraint
    #def constraint_dual(model_2):
     #   return (model_2.V1_t[25] - model_2.I_2 + model_2.Q_t[25]== model_2.x_1) #legger til at output volum fra masterproblem er nå input volum i sub problem
    #model_2.C6 = pyo.Constraint(rule=constraint_dual)

    def constraint_dual(model_2):
        return(model_2.SV == model_2.x_1)
    model_2.C6 = pyo.Constraint(rule=constraint_dual)

    def constraint_V3(model_2,s):
        return (model_2.V1_ts[25,s] == model_2.SV + model_2.I_2*s - model_2.Q_ts[25,s]) #sjekk denne mer
    model_2.C7 = pyo.Constraint(model_2.S,rule=constraint_V3)

    # constraint 1, ensure that Q_t is lower than Qmax
    def constraint_Q(model_2, t,s):
        return model_2.Q_ts[t,s] <= model_2.Q_Max
    model_2.C1 = pyo.Constraint(model_2.T_2,model_2.S, rule=constraint_Q)

     # constraint 2, constraint for water volume
    def constraint_V(model_2, t,s):
        if (t>=26):
            return (model_2.V1_ts[t,s] == model_2.V1_ts[t - 1,s] + model_2.I_2*s - model_2.Q_ts[t,s])
        else:
            return pyo.Constraint.Skip
    model_2.C2 = pyo.Constraint(model_2.T_2,model_2.S, rule=constraint_V)

    # Constraint 3, ensure that V_t is lower than Vmax
    def constraint_V2(model_2, t,s):
        return (model_2.V1_ts[t,s] <= model_2.V_MAX)
    model_2.C3 = pyo.Constraint(model_2.T_2,model_2.S, rule=constraint_V2)

    # Constraint 4, P_vt, kan legges i obj
    def constraint_P1(model_2, t,s):
         return (model_2.P_ts[t,s] == model_2.Q_ts[t,s] * model_2.E_conv * (1 / model_2.M_conv))
    model_2.C4 = pyo.Constraint(model_2.T_2,model_2.S, rule=constraint_P1)

    # Constraint 5, ensure that P_ts is lower than Pmax, trenger egentlig ikke
    def constraint_Pmax(model_2, t,s):
         return (model_2.P_ts[t,s] <= model_2.P_Max)
    model_2.C5 = pyo.Constraint(model_2.T_2,model_2.S, rule=constraint_Pmax)

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


def FirstDayProblem(Cuts_data):  # ha med itaration,
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
    object = pyo.value(model.obj_del1)
    print("dette er verdien til obj:", object)
    return object




def CreateCuts(HjelpeListe):
    c=0
    d=0
    for i in range(5):
        c = c + HjelpeListe[i]["dual:"]
        d = d + HjelpeListe[i]["obj2:"] - HjelpeListe[i]["dual:"]*HjelpeListe[i]["x_1"]
    a = c/5
    b = d/5
    return (a,b)


DiscreteList = [0,1,2,3,4,5,6,7,8,9]
Cuts_data = {}
ListOfCuts = []
HjelpeListe = []

for it in range(10):

    for s in range(5):
        HjelpeListe.append(s)
        HjelpeListe[s] = {}
        obj_2, dual = SubProblem(DiscreteList[it])
        print("OBJ2", obj_2, "dual:",dual)
        HjelpeListe[s] = {"obj2:": obj_2, "dual:": dual, "x_1": DiscreteList[it]}

    a, b = CreateCuts(HjelpeListe)
    print("it:", it, "a: ", a, "b:",b)


    Cuts_data[it] = {}
    Cuts_data[it]["slope"] = a
    Cuts_data[it]["constant"] = b

print(Cuts_data)
object = FirstDayProblem(Cuts_data)
print("obj har verdi:", object)
print("test")














