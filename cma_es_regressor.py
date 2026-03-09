from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
import cma
import torch
from models.PINN_inter_model import PINN
import math

training_data = pd.read_csv("csv/output.csv")

# X = training_data[training_data["freq"] == 50.5]

X = training_data.to_numpy()

X_train = X[:,:7]

R = X[:,8]
L = X[:,9]

validation2 = pd.read_csv('csv/output.csv').to_numpy()



R_r = TabPFNRegressor(device="cuda")
L_r = TabPFNRegressor(device="cuda")



R_r.fit(validation2[:,:7], validation2[:,8])
L_r.fit(validation2[:,:7], validation2[:,9])


model = PINN()
model.load_state_dict(torch.load('models/saved/PINN_para_model.pth'))


target_freq = 75.25


# --------- 设计参数约束（常见：工艺/几何范围） ---------
# 这里假设每个设计参数都在 [-2, 2]
lb = np.array([10, 200,  100,8, 15, 2])
ub = np.array([20,250, 350, 16, 20, 6])


def normalize(x):
    return (x - lb) / (ub - lb)

def denormalize(xn):
    return lb + xn * (ub - lb)





# --------- CMA-ES 设置 ---------
x0 = normalize(np.array([15, 225,  225,12, 17, 4]))     # 初始均值（可用经验值）
sigma0 = 0.2         # 初始步长（搜索尺度，和变量范围同量纲）



def rounding(x):
    x = x.reshape(-1)
    v_lin = denormalize(x)
    v_int = np.round(v_lin)
    return v_int

# --------- 你的正向模型：x(6,) -> y(2,) ---------

def forward(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(1,-1)

    # x = denormalize(x)
    print(x)
    new_col = np.full((x.shape[0], 1), target_freq)
    x = np.hstack((x, new_col))
    # print(x.shape)


    ######### TabPFN

    ty1 = R_r.predict(x)
    ty2 = L_r.predict(x)

    ######## PINN
    x = torch.from_numpy(np.log(x)).float()

    y1, y2 = np.exp(model(x[:,:6],x[:,6]).T.detach().numpy())


    return np.array([y1, y2, ty1, ty2], dtype=np.float64)

# --------- 逆向目标：给定 2 维目标指标 ---------
y_target = np.array([300, 4,300,4], dtype=np.float64)



# --------- 代价函数：让 forward(x) 接近 y_target ---------
# 可加权、可加惩罚、可加正则（例如偏好更小的参数）
w = np.array([1,1,0.6,0.6], dtype=np.float64)  # 两个指标的权重

def objective(x: np.ndarray) -> float:

    x = np.asarray(x, dtype=np.float64)


    #
    x = rounding(x)

    # 简单的边界惩罚（也可用 cma 的边界处理，但惩罚法更直观）
    penalty = 0.0


    print("==========++++++++++++++++++++++++++++==========")



    y = forward(x).reshape(-1)

    print(y)

    err = (y - y_target) * w
    loss = float(np.dot(err, err))
    print("loss: ", loss)
    # 可选：让解更“温和”的正则（按需打开）
    # loss += 1e-3 * float(np.dot(x, x))

    return loss + penalty


opts = {
    "popsize": 24,                 # 种群大小（维度6通常 16~40 都行）
    "maxiter": 2000,                # 最大迭代
    "tolfun": 4e-1,               # 收敛阈值
    "tolx": 4e-1,
    "verb_disp": 20,               # 每隔多少代打印一次
    "ftarget": 10,
    "bounds": [0, 1],
    "tolflatfitness": 50,
    "tolstagnation": 300,
    # 若你希望更强的边界处理，也可以用：
    # "bounds": [lb.tolist(), ub.tolist()],
}

flag = 0



es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

while not es.stop():
    X = es.ask()                           # 采样一批候选 x
    F = [objective(x) for x in X]          # 评估目标函数
    es.tell(X, F)                          # 更新分布
    es.disp()



res = es.result
x_best = np.array(res.xbest, dtype=np.float64)



y_best = forward(rounding(x_best))
final_loss = objective(x_best)

x_best = rounding(x_best)

print("\n===== RESULT =====")
print("y_target =", y_target)
print("x_best   =", x_best)
print("y_best   =", y_best)
print("loss     =", final_loss)
print("stop for", es.stop())


# print(df.columns)

print("=================")
print("ANN value: ", forward(x_best))


df = pd.read_csv("csv/cma.csv")


new_roll = {
"tCu": x_best[0], "wCu": x_best[1], "tLam": x_best[2], "nLam": x_best[3],
             "aln": x_best[4], "tsu": x_best[5], "freq": target_freq
}
df.loc[len(df)] = new_roll
df.to_csv("csv/cma.csv", index=False)