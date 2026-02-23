from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from tabpfn_extensions.rf_pfn import RandomForestTabPFNRegressor
from tabpfn_extensions import interpretability
from tabpfn import TabPFNRegressor
from tabpfn import TabPFNClassifier
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
import torch
from tabpfn.constants import ModelVersion
import silhosue_kmean
import pandas as pd
import numpy as np


def test(X, targetR, targetL, f):
    # train_data = pd.read_csv('../ML_Inductor_QLR_Predictor/training_csv/interpolation_data.csv').to_numpy()
    # test_data = pd.read_csv("../ML_Inductor_QLR_Predictor/training_csv/pinn_data.csv").to_numpy()
    train_data = pd.read_csv("../ML_Inductor_QLR_Predictor/training_csv/pinn_data.csv").to_numpy()
    test_data = pd.read_csv('../ML_Inductor_QLR_Predictor/training_csv/pinn_data.csv').to_numpy()

    X_train = train_data[:,:7]
    y_train_R = train_data[:,8]
    y_train_L = train_data[:,9]

    X_test= test_data[:,:7]
    y_test_R = test_data[:,8]
    y_test_L = test_data[:,9]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


    R_regressor = TabPFNRegressor()

    L_regressor = TabPFNRegressor()

    R_regressor.fit(X_train, y_train_R)
    L_regressor.fit(X_train, y_train_L)


    print(X.shape)
    print(X_train.shape)

    r = R_regressor.predict(X)
    l = L_regressor.predict(X)


    print(r, l)
    designs = []
    freq = [f]
    print("===========Validating===========")
    for i in range(len(X)):
        temp = []
        if r[i] <= targetR*1.1 and r[i] >= targetR*0.9 and l[i] <= targetL*1.1 and l[i] >= targetL*0.9:
            print(i)
            temp = np.concatenate((X[i].reshape(-1), np.array([r[i], l[i]])))
            designs.append(temp)

    print("=========Validated Designs==========")
    print(designs)
    print(len(designs))
    designs = np.array(designs)

    output_df = pd.DataFrame(
        {"tCu": designs[:, 0], "wCu": designs[:, 1], "tLam": designs[:, 2], "nLam": designs[:, 3],
         "aln": designs[:, 4], "tsu": designs[:, 5], "freq": designs[:, 6],
         "Pre_R": designs[:,7], "Pre_L": designs[:,8]})


    output_df.to_csv("csv/designs.csv", index=False)