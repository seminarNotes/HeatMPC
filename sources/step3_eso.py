# -*- coding: utf-8 -*-

import numpy as np


# ----------------------------------------------------
# ESO L gain 계산
# ----------------------------------------------------
def eso_gain(a, p1, p2):
    p1 = float(p1)
    p2 = float(p2)

    l1 = a + 1.0 - (p1 + p2)
    l2 = p1 * p2 - (p1 + p2) + 1.0

    L = np.array([[l1], [l2]], dtype=float)
    A = np.array([[a, 1.0],
                  [0.0, 1.0]], dtype=float)
    C = np.array([[1.0, 0.0]], dtype=float)
    Acl = A - L @ C
    poles = np.linalg.eigvals(Acl)

    return L, Acl, poles


# ----------------------------------------------------
# ESO 초기화
# ----------------------------------------------------
def eso_reset(T0, z0=0.0):
    return np.array([[float(T0)], [float(z0)]], dtype=float)


# ----------------------------------------------------
# ESO 1-step 업데이트
# ----------------------------------------------------
def eso_step(T_meas, u,
             x_hat,
             a, b,
             L):
    T_meas = float(T_meas)
    u = float(u)

    A = np.array([[a, 1.0],
                  [0.0, 1.0]])
    B = np.array([[b],
                  [0.0]])
    C = np.array([[1.0, 0.0]])

    x_pred = A @ x_hat + B * u

    y_hat = float(C @ x_hat)
    e = T_meas - y_hat

    x_hat_next = x_pred + L * e
    return x_hat_next


# ----------------------------------------------------
# ESO 현재 추정값 반환
# ----------------------------------------------------
def eso_get_state(x_hat):
    return float(x_hat[0, 0]), float(x_hat[1, 0])


# ----------------------------------------------------
# 간단 테스트
# ----------------------------------------------------
if __name__ == "__main__":
    a = 0.99895
    b = 0.949229

    L, Acl, poles = eso_gain(a, 0.4, 0.5)

    print("=== ESO TEST ===")
    print("L =")
    print(L)
    print(" poles =", poles)

    x_hat = eso_reset(400.0, 0.0)

    T_true = 400.0
    z_true = -5.0
    u_k = 0.5

    print("\nstep | T_true   T_hat    z_hat")
    print("--------------------------------")
    for k in range(50):
        T_true = a * T_true + b * u_k + z_true
        x_hat = eso_step(T_true, u_k, x_hat, a, b, L)
        T_hat, z_hat = eso_get_state(x_hat)
        print(f"{k:4d} | {T_true:7.2f}  {T_hat:7.2f}  {z_hat:7.2f}")
