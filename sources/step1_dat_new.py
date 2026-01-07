import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# CSV 읽기
df = pd.read_csv("/home/junhuiwoo/Desktop/HeatMPC/data/targeted_1213.csv")

# 온도와 전력
T = df["온도"].values
power = df["전력량(1분)"].values

# u_k = 전력량 * 3/5
u = power * (3/5)

# -----------------------------
# 시프트 (k -> k+1)
# -----------------------------
T_k = T[:-1]          # T_k
u_k = u[:-1]          # u_k
T_k1 = T[1:]          # T_{k+1}

# 회귀용 행렬
X = np.column_stack([T_k, u_k])
y = T_k1

# 선형회귀 (절편 없음이 물리적으로 더 자연스러움)
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

a, b = model.coef_

print(f"T[k+1] = {a:.6f} * T[k] + {b:.6f} * u[k]")
print(df)