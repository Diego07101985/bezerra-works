import matplotlib.pyplot as plt
from utils import load_data, fetch_data, save_fig
from train_test import split_train_test_by_id
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from utils import load_data
import os
import pandas as pd


HOUSING_PATH = os.path.join("datasets", "housing")


housing = load_data(HOUSING_PATH, "housing.csv")

housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

print(test_set.head())


housing["median_income"].hist()
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts()
housing["income_cat"].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = split_train_test_by_id(
    housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()

compare_props["Rand. %error"] = 100 * \
    compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * \
    compare_props["Stratified"] / compare_props["Overall"] - 100
