from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
import cma
import torch
from models.PINN_inter_model import PINN
import math

training_data = pd.read_csv("csv/output.csv")

X = training_data[training_data["freq"] == 50.5]
X_train = X.iloc[:,0:7]

R = X.iloc[:,8]
L = X.iloc[:,9]

R_r = TabPFNRegressor(device="cuda")
L_r = TabPFNRegressor(device="cuda")

print(X.shape,X_train.shape, R.shape, L.shape)


R_r.fit(X_train.to_numpy(), R.to_numpy())
L_r.fit(X_train.to_numpy(), L.to_numpy())


model = PINN()
model.load_state_dict(torch.load('models/saved/PINN_para_model.pth'))

def rounding(x):
    v_lin = float(math.exp(x))  # log域 -> 线性域 (float)
    v_int = int(np.round(v_lin))  # 量化为整数
    return float(math.log(v_int))  # 再回到 log域 (float)

# --------- 你的正向模型：x(6,) -> y(2,) ---------

def forward(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(1,-1)
    new_col = np.full((x.shape[0], 1), np.log(50.5))
    x = np.hstack((x, new_col))
    # print(x.shape)

    ######### TabPFN
    y1 = R_r.predict(np.exp(x))
    y2 = L_r.predict(np.exp(x))

    ######## PINN
    # x = torch.from_numpy(x).float()

    # y1, y2 = np.exp(model(x[:,:6],x[:,6]).T.detach().numpy())




    return np.array([y1, y2], dtype=float)

# --------- 逆向目标：给定 2 维目标指标 ---------
y_target = np.array([150, 5], dtype=float)

# --------- 设计参数约束（常见：工艺/几何范围） ---------
# 这里假设每个设计参数都在 [-2, 2]
lb = np.log([10, 200,  100,8, 15, 2])
ub =  np.log([20,250, 350, 16, 20, 6])

# --------- 代价函数：让 forward(x) 接近 y_target ---------
# 可加权、可加惩罚、可加正则（例如偏好更小的参数）
w = np.array([1.0, 10], dtype=float)  # 两个指标的权重

def objective(x: np.ndarray) -> float:

    x = np.asarray(x, dtype=float)


    #
    x[0] = rounding(x[0])
    x[1] = rounding(x[1])
    x[2] = rounding(x[2])
    x[3] = rounding(x[3])
    x[4] = rounding(x[4])
    x[5] = rounding(x[5])

    # 简单的边界惩罚（也可用 cma 的边界处理，但惩罚法更直观）
    penalty = 0.0
    # if np.any(x < lb) or np.any(x > ub):
    #     rng = (ub - lb)
    #     v = (np.maximum(lb - x, 0.0) + np.maximum(x - ub, 0.0)) / rng
    #     penalty = 10 * float(np.dot(v, v))

    y = forward(x).reshape(-1)
    # print(f"y shape{y.shape}")

    err = (y - y_target) * w
    loss = float(np.dot(err, err))

    # 可选：让解更“温和”的正则（按需打开）
    # loss += 1e-3 * float(np.dot(x, x))

    return loss + penalty

# --------- CMA-ES 设置 ---------
x0 = np.log([10, 200,  100,8, 15, 2])       # 初始均值（可用经验值）
sigma0 = 0.3         # 初始步长（搜索尺度，和变量范围同量纲）

opts = {
    "popsize": 24,                 # 种群大小（维度6通常 16~40 都行）
    "maxiter": 200,                # 最大迭代
    "tolfun": 2e-1,               # 收敛阈值
    "tolx": 2e-1,
    "verb_disp": 20,               # 每隔多少代打印一次
    "ftarget": 2e-1,
    "bounds": [lb, ub],
    # 若你希望更强的边界处理，也可以用：
    # "bounds": [lb.tolist(), ub.tolist()],
}

es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

while not es.stop():
    X = es.ask()                           # 采样一批候选 x
    F = [objective(x) for x in X]          # 评估目标函数
    es.tell(X, F)                          # 更新分布
    es.disp()

res = es.result
x_best = np.array(res.xbest, dtype=float)
y_best = forward(x_best)
final_loss = objective(x_best)

x_best = np.exp(x_best)

print("\n===== RESULT =====")
print("y_target =", y_target)
print("x_best   =", x_best)
print("y_best   =", y_best)
print("loss     =", final_loss)


df = pd.read_csv("csv/cma.csv")

# print(df.columns)

new_roll = {
"tCu": x_best[0], "wCu": x_best[1], "tLam": x_best[2], "nLam": x_best[3],
             "aln": x_best[4], "tsu": x_best[5], "freq": 50.5
}
df.loc[len(df)] = new_roll
df.to_csv("csv/cma.csv", index=False)