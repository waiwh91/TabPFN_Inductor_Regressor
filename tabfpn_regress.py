from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from tabpfn_extensions.rf_pfn import RandomForestTabPFNRegressor
from tabpfn import TabPFNRegressor
from tabpfn import TabPFNClassifier
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
import torch
from tabpfn.constants import ModelVersion
import pandas as pd
import numpy as np

train_data = pd.read_csv('../ML_Inductor_QLR_Predictor/training_csv/interpolation_data.csv').to_numpy()
test_data = pd.read_csv("../ML_Inductor_QLR_Predictor/training_csv/pinn_data.csv").to_numpy()
X_train = train_data[:,:7]
y_train_R = train_data[:,8]
y_train_L = train_data[:,9]

X_test= test_data[:,:7]
y_test_R = test_data[:,8]
y_test_L = test_data[:,9]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


R_reg = TabPFNRegressor()
R_regressor = RandomForestTabPFNRegressor(tabpfn=R_reg)
L_reg = TabPFNRegressor()
L_regressor = RandomForestTabPFNRegressor(tabpfn=L_reg)
R_regressor.fit(X_train, y_train_R)
L_regressor.fit(X_train, y_train_L)


R_predictions = R_regressor.predict(X_test)
R_mpe = mean_absolute_percentage_error(y_test_R, R_predictions)
R_r2 = r2_score(y_test_R, R_predictions)

print("Mean percentage Error (MPE) for R:", R_mpe)
print("R² Score:", R_r2)

L_predictions = L_regressor.predict(X_test)
L_mpe = mean_absolute_percentage_error(y_test_L, L_predictions)
L_r2 = r2_score(y_test_L, L_predictions)

print("Mean percentage Error (MPE) for L:", L_mpe)
print("R² Score:", L_r2)

q = 2 * np.pi * X_test[:,6] * L_predictions / R_predictions




output_df = pd.DataFrame(
        {"tCu": X_test[:, 0], "wCu": X_test[:, 1], "tLam": X_test[:, 2], "nLam": X_test[:, 3],
             "aln": X_test[:, 4], "tsu": X_test[:, 5], "freq": X_test[:,6], "Pre_Q": q,
             "Pre_R": R_predictions, "Pre_L": L_predictions})



output_df.to_csv("output.csv", index=False)

