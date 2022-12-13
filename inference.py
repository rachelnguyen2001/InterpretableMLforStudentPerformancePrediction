import random
import pandas as pd
import math
import numpy as np
import copy
from sklearn.ensemble import RandomForestRegressor
from models import MSE

def process_data(Xmat):
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
    
    l = len(Xmat)
    Mjob_other, Mjob_athome, Mjob_civil, Mjob_health, Mjob_teacher = [None]*l, [None]*l, [None]*l, [None]*l, [None]*l
    Fjob_other, Fjob_athome, Fjob_civil, Fjob_health, Fjob_teacher = [None]*l, [None]*l, [None]*l, [None]*l, [None]*l
    guard_mom, guard_fat, guard_other = [None]*l, [None]*l, [None]*l
    reason_home, reason_rep, reason_course, reason_other = [None]*l, [None]*l, [None]*l, [None]*l
    i = 0

    for ind in Xmat.index:
        Mjob = Xmat["Mjob"][ind]

        if Mjob == "at_home":
            Mjob_athome[i], Mjob_other[i], Mjob_civil[i], Mjob_health[i], Mjob_teacher[i] = 1, 0, 0, 0, 0
        elif Mjob == "health":
            Mjob_athome[i], Mjob_other[i], Mjob_civil[i], Mjob_health[i], Mjob_teacher[i] = 0, 0, 0, 1, 0
        elif Mjob == "services":
            Mjob_athome[i], Mjob_other[i], Mjob_civil[i], Mjob_health[i], Mjob_teacher[i] = 0, 0, 1, 0, 0
        elif Mjob == "teacher":
            Mjob_athome[i], Mjob_other[i], Mjob_civil[i], Mjob_health[i], Mjob_teacher[i] = 0, 0, 0, 0, 1
        else:
            Mjob_athome[i], Mjob_other[i], Mjob_civil[i], Mjob_health[i], Mjob_teacher[i] = 0, 1, 0, 0, 0
    
        Fjob = Xmat["Fjob"][ind]

        if Fjob == "at_home":
            Fjob_athome[i], Fjob_other[i], Fjob_civil[i], Fjob_health[i], Fjob_teacher[i] = 1, 0, 0, 0, 0
        elif Fjob == "health":
            Fjob_athome[i], Fjob_other[i], Fjob_civil[i], Fjob_health[i], Fjob_teacher[i] = 0, 0, 0, 1, 0
        elif Fjob == "services":
            Fjob_athome[i], Fjob_other[i], Fjob_civil[i], Fjob_health[i], Fjob_teacher[i] = 0, 0, 1, 0, 0
        elif Fjob == "teacher":
            Fjob_athome[i], Fjob_other[i], Fjob_civil[i], Fjob_health[i], Fjob_teacher[i] = 0, 0, 0, 0, 1
        else:
            Fjob_athome[i], Fjob_other[i], Fjob_civil[i], Fjob_health[i], Fjob_teacher[i] = 0, 1, 0, 0, 0
    
        guard = Xmat["guardian"][ind]
        if guard == "mother":
            guard_mom[i], guard_fat[i], guard_other[i] = 1, 0, 0
        elif guard == "father":
            guard_mom[i], guard_fat[i], guard_other[i] = 0, 1, 0
        else:
            guard_mom[i], guard_fat[i], guard_other[i] = 0, 0, 1

        reason = Xmat["reason"][ind]
        if reason == "home":
            reason_home[i], reason_rep[i], reason_course[i], reason_other[i] = 1, 0, 0, 0
        elif reason == "reputation":
            reason_home[i], reason_rep[i], reason_course[i], reason_other[i] = 0, 1, 0, 0
        elif reason == "course":
            reason_home[i], reason_rep[i], reason_course[i], reason_other[i] = 0, 0, 1, 0
        else:
            reason_home[i], reason_rep[i], reason_course[i], reason_other[i] = 0, 0, 0, 1

        i += 1
    
    Xmat["Mjob_other"], Xmat["Mjob_athome"], Xmat["Mjob_civil"], Xmat["Mjob_health"], Xmat["Mjob_teacher"] = Mjob_other, Mjob_athome, Mjob_civil, Mjob_health, Mjob_teacher
    Xmat = Xmat.drop(["Mjob"], axis="columns")

    Xmat["Fjob_other"], Xmat["Fjob_athome"], Xmat["Fjob_civil"], Xmat["Fjob_health"], Xmat["Fjob_teacher"] = Fjob_other, Fjob_athome, Fjob_civil, Fjob_health, Fjob_teacher
    Xmat = Xmat.drop(["Fjob"], axis="columns")

    Xmat["guard_mom"], Xmat["guard_fat"], Xmat["guard_other"] = guard_mom, guard_fat, guard_other
    Xmat = Xmat.drop(["guardian"], axis="columns")

    Xmat["reason_home"], Xmat["reason_rep"], Xmat["reason_course"], Xmat["reason_other"] = reason_home, reason_rep, reason_course, reason_other
    Xmat = Xmat.drop(["reason"], axis="columns")
    
    return Xmat

# Get a minipatch of size n with m features
def get_minipatch(data, n, m):
    columns = list(data.columns)
    columns.remove("G3")
    
    to_keep_col = random.sample(columns, m)
    to_remove_col = copy.deepcopy(columns)

    for name in columns:
        if name in to_keep_col:
            to_remove_col.remove(name)
    
    data = data.drop(to_remove_col, axis="columns")
    
    row_index = [i for i in range(len(data))]
    to_keep_row = random.sample(row_index, n)
    for i in range(len(data)):
        if i in to_keep_row:
            row_index.remove(i)

    data = data.drop(row_index)
    Xmat = data.drop(["G3"], axis="columns")
    Y = np.array([outcome for outcome in data["G3"]])
    return Xmat, Y, to_keep_row, to_remove_col

# Get a minipatch of size n with m features such that col column is preserved
def get_minipatch_with_col(data, n, m, col):
    columns = list(data.columns)
    columns.remove("G3")
    columns.remove(col)
    
    to_keep_col = random.sample(columns, m-1)
    columns.append(col)
    to_keep_col.append(col)
    to_remove_col = copy.deepcopy(columns)

    for name in columns:
        if name in to_keep_col:
            to_remove_col.remove(name)
    
    data = data.drop(to_remove_col, axis="columns")
    
    row_index = [i for i in range(len(data))]
    to_keep_row = random.sample(row_index, n)
    for i in range(len(data)):
        if i in to_keep_row:
            row_index.remove(i)

    data = data.drop(row_index)
    Xmat = data.drop(["G3"], axis="columns")
    Y = np.array([outcome for outcome in data["G3"]])
    return Xmat, Y, to_keep_row, to_remove_col

# Get LOO error
def get_LOO_err(patches, X, Y, i):
    Yhat = [0]
    counter = 0
    for (_, _, to_keep_row, to_remove_col, model) in patches:
        if i not in to_keep_row:
            X_selected = X.drop(to_remove_col, axis="columns")
            counter += 1
            Yhat += model.predict(X_selected)

    if counter == 0:
        return None

    Yhat[0] /= counter
    return MSE(Y, Yhat)

# Get LOO + LOCO error
def get_LOO_and_LOCO_err(patches, X, Y, i, col):
    Yhat = [0]
    counter = 0

    for (Xmat, _, to_keep_row, to_remove_col, model) in patches:
        if i not in to_keep_row and col not in list(Xmat.columns):
            counter += 1
            X_selected = X.drop(to_remove_col, axis="columns")
            Yhat += model.predict(X_selected)
    
    if counter == 0:
        return None

    Yhat[0] /= counter
    return MSE(Y, Yhat)

# Calculate confidence interval for feature importance score
def inference(data, col, n, m, K, z):
    patches = []

    # Create minipatch without the col feature
    for k in range(5):
        Xmat, Y, to_keep_row, to_remove_col = get_minipatch(data.drop([col], axis="columns"), n, m)
        to_remove_col.append(col)
        model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        model.fit(Xmat, Y)
        patches.append((Xmat, Y, to_keep_row, to_remove_col, model))
    
    # Create minipatch with the col feature
    for k in range(5, 10):
        Xmat, Y, to_keep_row, to_remove_col = get_minipatch_with_col(data, n, m, col)
        model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        model.fit(Xmat, Y)
        patches.append((Xmat, Y, to_keep_row, to_remove_col, model))
    
    # Create random minipatch
    for k in range(10, K):
        Xmat, Y, to_keep_row, to_remove_col = get_minipatch(data, n, m)
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(Xmat, Y)
        patches.append((Xmat, Y, to_keep_row, to_remove_col, model))

    # Calculate the mean
    mean = 0
    std = 0
    N = len(data)
    Xmat = data.drop(["G3"], axis="columns")
    Y = np.array([outcome for outcome in data["G3"]])
    num_samples = 0

    for i in range(N):
        row = data.iloc[i].to_frame().transpose()
        X = row.drop(["G3"], axis="columns")
        Y = np.array([outcome for outcome in row["G3"]])
        loo = get_LOO_err(patches, X, Y, i)
        loo_and_loco = get_LOO_and_LOCO_err(patches, X, Y, i, col)

        if loo is not None and loo_and_loco is not None:
            mean += (loo - loo_and_loco)
            num_samples += 1
        
    mean /= num_samples

    # Calculate the standard deviation
    num_samples = 0
    for i in range(N):
        row = data.iloc[i].to_frame().transpose()
        X = row.drop(["G3"], axis="columns")
        Y = np.array([outcome for outcome in row["G3"]])
        loo = get_LOO_err(patches, X, Y, i)
        loo_and_loco = get_LOO_and_LOCO_err(patches, X, Y, i, col)

        if loo is not None and loo_and_loco is not None:
            std += (loo - loo_and_loco - mean)**2
            num_samples += 1
        
    std /= (num_samples - 1)
    std = math.sqrt(std)

    # Confidence interval
    start = mean - (z * std) / math.sqrt(num_samples)
    end = mean + (z * std) / math.sqrt(num_samples)
    return start, end



def main():
    random.seed(42)

    # Mathematics
    print("Results for mathematics class: ")
    data = pd.read_table("data/student-mat.csv", sep=";")
    data = process_data(data)
    columns = list(data.columns)
    columns.remove("G3")

    for name in columns:
        start, end = inference(data, name, 30, 5, 15, 1.96)
        print("95% CI for feature importance score of {var} is: [{start}, {end}]".format(var=name, start=start, end=end))
    
    # Portugese
    print("\nResults for Portugese class: ")
    data = pd.read_table("data/student-por.csv", sep=";")
    data = process_data(data)
    columns = list(data.columns)
    columns.remove("G3")

    for name in columns:
        start, end = inference(data, name, 50, 5, 15, 1.96)
        print("95% CI for feature importance score of {var} is: [{start}, {end}]".format(var=name, start=start, end=end))
    
if __name__ == "__main__":
    main()