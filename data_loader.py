import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filename):
    data = pd.read_table(filename, sep=";")
    Xmat = data.drop(["G3"], axis="columns")
    Y = np.array([outcome for outcome in data["G3"]])

    Xmat["sex"] = Xmat["sex"].replace("M", 1)
    Xmat["sex"] = Xmat["sex"].replace("F", 0)
    
    Xmat["school"] = Xmat["school"].replace("GP", 1)
    Xmat["school"] = Xmat["school"].replace("MS", 0)

    Xmat["address"] = Xmat["address"].replace("U", 1)
    Xmat["address"] = Xmat["address"].replace("R", 0)

    Xmat["famsize"] = Xmat["famsize"].replace("GT3", 1)
    Xmat["famsize"] = Xmat["famsize"].replace("LE3", 0)
    
    Xmat["Pstatus"] = Xmat["Pstatus"].replace("T", 1)
    Xmat["Pstatus"] = Xmat["Pstatus"].replace("A", 0)

    yes_no_var = ["schoolsup", "famsup", "activities", "paid", "internet", "nursery", "higher", "romantic"]

    for var in yes_no_var:
        Xmat[var] = Xmat[var].replace("yes", 1)
        Xmat[var] = Xmat[var].replace("no", 0)

    n = len(Xmat)
    Mjob_other, Mjob_athome, Mjob_civil, Mjob_health, Mjob_teacher = [None]*n, [None]*n, [None]*n, [None]*n, [None]*n
    Fjob_other, Fjob_athome, Fjob_civil, Fjob_health, Fjob_teacher = [None]*n, [None]*n, [None]*n, [None]*n, [None]*n
    guard_mom, guard_fat, guard_other = [None]*n, [None]*n, [None]*n
    reason_home, reason_rep, reason_course, reason_other = [None]*n, [None]*n, [None]*n, [None]*n
    
    for ind in Xmat.index:
        Mjob = Xmat["Mjob"][ind]

        if Mjob == "at_home":
            Mjob_athome[ind], Mjob_other[ind], Mjob_civil[ind], Mjob_health[ind], Mjob_teacher[ind] = 1, 0, 0, 0, 0
        elif Mjob == "health":
            Mjob_athome[ind], Mjob_other[ind], Mjob_civil[ind], Mjob_health[ind], Mjob_teacher[ind] = 0, 0, 0, 1, 0
        elif Mjob == "services":
            Mjob_athome[ind], Mjob_other[ind], Mjob_civil[ind], Mjob_health[ind], Mjob_teacher[ind] = 0, 0, 1, 0, 0
        elif Mjob == "teacher":
            Mjob_athome[ind], Mjob_other[ind], Mjob_civil[ind], Mjob_health[ind], Mjob_teacher[ind] = 0, 0, 0, 0, 1
        else:
            Mjob_athome[ind], Mjob_other[ind], Mjob_civil[ind], Mjob_health[ind], Mjob_teacher[ind] = 0, 1, 0, 0, 0
        
        Fjob = Xmat["Fjob"][ind]
        if Fjob == "at_home":
            Fjob_athome[ind], Fjob_other[ind], Fjob_civil[ind], Fjob_health[ind], Fjob_teacher[ind] = 1, 0, 0, 0, 0
        elif Fjob == "health":
            Fjob_athome[ind], Fjob_other[ind], Fjob_civil[ind], Fjob_health[ind], Fjob_teacher[ind] = 0, 0, 0, 1, 0
        elif Fjob == "services":
            Fjob_athome[ind], Fjob_other[ind], Fjob_civil[ind], Fjob_health[ind], Fjob_teacher[ind] = 0, 0, 1, 0, 0
        elif Fjob == "teacher":
            Fjob_athome[ind], Fjob_other[ind], Fjob_civil[ind], Fjob_health[ind], Fjob_teacher[ind] = 0, 0, 0, 0, 1
        else:
            Fjob_athome[ind], Fjob_other[ind], Fjob_civil[ind], Fjob_health[ind], Fjob_teacher[ind] = 0, 1, 0, 0, 0
        
        guard = Xmat["guardian"][ind]
        if guard == "mother":
            guard_mom[ind], guard_fat[ind], guard_other[ind] = 1, 0, 0
        elif guard == "father":
            guard_mom[ind], guard_fat[ind], guard_other[ind] = 0, 1, 0
        else:
            guard_mom[ind], guard_fat[ind], guard_other[ind] = 0, 0, 1

        reason = Xmat["reason"][ind]
        if reason == "home":
            reason_home[ind], reason_rep[ind], reason_course[ind], reason_other[ind] = 1, 0, 0, 0
        elif reason == "reputation":
            reason_home[ind], reason_rep[ind], reason_course[ind], reason_other[ind] = 0, 1, 0, 0
        elif reason == "course":
            reason_home[ind], reason_rep[ind], reason_course[ind], reason_other[ind] = 0, 0, 1, 0
        else:
            reason_home[ind], reason_rep[ind], reason_course[ind], reason_other[ind] = 0, 0, 0, 1

    Xmat["Mjob_other"], Xmat["Mjob_athome"], Xmat["Mjob_civil"], Xmat["Mjob_health"], Xmat["Mjob_teacher"] = Mjob_other, Mjob_athome, Mjob_civil, Mjob_health, Mjob_teacher
    Xmat["Fjob_other"], Xmat["Fjob_athome"], Xmat["Fjob_civil"], Xmat["Fjob_health"], Xmat["Fjob_teacher"] = Fjob_other, Fjob_athome, Fjob_civil, Fjob_health, Fjob_teacher
    Xmat["guard_mom"], Xmat["guard_fat"], Xmat["guard_other"] = guard_mom, guard_fat, guard_other
    Xmat["reason_home"], Xmat["reason_rep"], Xmat["reason_course"], Xmat["reason_other"] = reason_home, reason_rep, reason_course, reason_other
    Xmat = Xmat.drop(["Mjob", "Fjob", "guardian", "reason"], axis="columns")

    Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.33, random_state=42)
    Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat_train, Y_train, test_size=0.33, random_state=42)
    
    mean_age = np.mean(Xmat_train["age"])
    std_age = np.std(Xmat_train["age"])
    Xmat_train["age"] = (Xmat_train["age"] - mean_age) / std_age
    Xmat_val["age"] = (Xmat_val["age"] - mean_age) / std_age
    Xmat_test["age"] = (Xmat_test["age"] - mean_age) / std_age

    mean_G1 = np.mean(Xmat_train["G1"])
    std_G1 = np.std(Xmat_train["G1"])
    Xmat_train["G1"] = (Xmat_train["G1"] - mean_G1) / std_G1
    Xmat_val["G1"] = (Xmat_val["G1"] - mean_G1) / std_G1
    Xmat_test["G1"] = (Xmat_test["G1"] - mean_G1) / std_G1

    mean_G2 = np.mean(Xmat_train["G2"])
    std_G2 = np.std(Xmat_train["G2"])
    Xmat_train["G2"] = (Xmat_train["G2"] - mean_G2) / std_G2
    Xmat_val["G2"] = (Xmat_val["G2"] - mean_G2) / std_G2
    Xmat_test["G2"] = (Xmat_test["G2"] - mean_G2) / std_G2

    mean_G3 = np.mean(Y_train)
    std_G3 = np.std(Y_train)
    Y_train = (Y_train - mean_G3) / std_G3
    Y_val = (Y_val - mean_G3) / std_G3
    Y_test = (Y_test - mean_G3) / std_G3

    return (Xmat_train, Xmat_val, Xmat_test, Y_train, Y_val, Y_test)
