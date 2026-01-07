# -*- coding: utf-8 -*-
"""
KNU4_mpc.py (T_ref_list 버전)

- 모든 참조는 시간열 T_ref_list 로부터 직접 가져옴
- 클래스 제거, 딕셔너리 제거
- 함수 기반
"""

import numpy as np
import cvxpy as cp

def compute_fallback_u_old(T_meas, u_prev, z_hat, a, b,
                       u_min, u_max, du_max, T_target=None):
    # 0) 유지하고 싶은 온도
    T_hold = float(T_target) if T_target is not None else float(T_meas)

    # 1) 유도된 유지 전력 u_eq
    if abs(b) > 1e-8:
        u_eq = ((1.0 - a) * T_hold - z_hat) / b
    else:
        # b ~ 0 이면 모델이 말이 안 되므로, 그냥 이전 입력 기준
        u_eq = u_prev

    # 2) bounds 안으로

    u_des = float(np.clip(u_eq, u_min, u_max))

    # 3) du_max 제약 고려
    lo = max(u_min, u_prev - du_max)
    hi = min(u_max, u_prev + du_max)

    if lo <= hi:
        # u_des를 [lo, hi]로 projection
        if u_des < lo:
            u_fb = lo
        elif u_des > hi:
            u_fb = hi
        else:
            u_fb = u_des
    else:
        # 모순일 때는 u_prev 기준으로
        u_fb = float(np.clip(u_prev, u_min, u_max))

    return u_fb

def compute_fallback_u(
    T_meas, u_prev, u_min, u_max, du_max,
    T_target=None, Kp=0.1,
):
    """
    MPC fail 시 사용하는 P 제어기 기반 fallback 입력 계산.

    u_fb = u_prev + Kp * (T_target - T_meas)
    이후:
      - [u_min, u_max]로 saturation
      - |u_fb - u_prev| <= du_max 제약으로 projection
    """
    T_meas  = float(T_meas)
    u_prev  = float(u_prev)

    # 0) 기준 온도: 주어진 T_target이 없으면 현재 온도를 유지하도록 설정
    T_hold = float(T_target) if T_target is not None else float(T_meas)

    # 1) P 제어 법칙: u_des = u_prev + Kp * (T_hold - T_meas)
    e_T   = T_hold - T_meas          # 온도 오차
    u_des = u_prev + Kp * e_T        # P 제어기 출력 (증분 형태)

    # 2) 1차로 입력 bounds 적용
    u_des = float(np.clip(u_des, u_min, u_max))

    # 3) du_max 제약 고려 (입력 변화율 제한)
    lo = max(u_min, u_prev - du_max)
    hi = min(u_max, u_prev + du_max)

    if lo <= hi:
        # u_des를 [lo, hi]로 projection
        if u_des < lo:
            u_fb = lo
        elif u_des > hi:
            u_fb = hi
        else:
            u_fb = u_des
    else:
        # 모순일 때는 u_prev를 bounds 안에 클리핑해서 사용
        u_fb = float(np.clip(u_prev, u_min, u_max))

    return u_fb


# ----------------------------------------------------
# 주요 MPC 함수 (T_ref_list 기반)
# ----------------------------------------------------
def solve_mpc(
    T_meas, u_prev, z_hat,
    k_global, T_ref_list,
    a, b,
    N,
    beta, lam, eta, gamma,
    u_min, u_max, du_max, u_peak,
    T_cap
):

    T_meas = float(T_meas)
    u_prev = float(u_prev)
    z_hat = float(z_hat)

    # horizon 내 참조열 구성
    Tref_horizon = []
    for i in range(N):
        idx = k_global + i
        if idx < len(T_ref_list):
            Tref_horizon.append(T_ref_list[idx])
        else:
            Tref_horizon.append(T_ref_list[-1])  # 마지막 값 유지

    # 의사결정변수 및 상태 리스트
    u = cp.Variable(N)
    T_list = [T_meas]

    cost = 0
    constraints = []

    for i in range(N):
        # ARX(1) 예측
        T_next = a * T_list[i] + b * u[i] + z_hat
        T_list.append(T_next)

        Tref_i = Tref_horizon[i]

        # 입력 변화 비용용 이전 입력
        u_im1 = u_prev if i == 0 else u[i - 1]

        # 비용함수
        cost += beta * cp.square(T_next - Tref_i)     # 트래킹
        cost += lam * cp.square(u[i] - u_im1)         # 입력 변화
        cost += eta * cp.square(u[i])                 # 입력 크기
        if gamma > 0:
            cost += gamma * cp.square(cp.pos(u[i] - u_peak))  # 피크 패널티

        # 제약조건
        T_floor = 0
        constraints += [
            u[i] >= u_min,
            u[i] <= u_max,
            u[i] - u_im1 <= du_max,
            u_im1 - u[i] <= du_max,
            T_next <= T_cap,
            T_next >= T_floor
        ]

    # QP 풀기
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        status = prob.status
    except:# Exception as e:
        # print(f"[MPC] Solver exception: {e}")
        status = "error"

    # 해가 없거나 최적이 아니면 → fallback 제어 사용
    if (status not in ["optimal", "optimal_inaccurate"]) or (u.value is None):
        # 현재/해당 시점 참조를 fallback 목표로 사용
        if len(T_ref_list) > 0:
            idx_now = min(k_global, len(T_ref_list) - 1)
            T_target = float(T_ref_list[idx_now])
        else:
            T_target = float(T_meas)

        u_fb = compute_fallback_u(
            T_meas=T_meas,
            u_prev=u_prev,
            u_min=u_min,
            u_max=u_max,
            du_max=du_max,
            T_target=T_target,
        )
        return u_fb

    # 정상적으로 풀린 경우: 첫 스텝 제어 입력 사용
    u0 = float(u.value[0])

    # 수치오차 방지용으로 한 번 더 bounds 클리핑
    u0 = float(np.clip(u0, u_min, u_max))
    return u0


def plot_tracking(T_log, T_ref_list):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(T_log, label="T")
    plt.plot(T_ref_list[:len(T_log)], label="T_ref", linestyle='--')
    plt.xlabel("time step k")
    plt.ylabel("Temperature")
    plt.title("MPC Tracking Result")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    a = 0.99895
    if True : 
        b = 1.449229
    else :
        b = 0.949229
    N = 10

    if False : 
        beta = 5.0
        lam = 0.1
        du_max = 0.2
        
    else : 

        beta = 0.8
        lam = 1.2
        du_max = 0.06
    
    u_max = 1.0
    eta = 0.0
    gamma = 0.0
    u_min = 0.0
    u_peak = 1.0

    T_cap = 1000.0
    dt = 1.0

    sim_len = 300
    T_ref_list = []
    for k in range(sim_len):
        if k < sim_len // 2:
            T_ref_list.append(930.0)
        else:
            T_ref_list.append(880.0)

    T = 900.0
    u_prev = 0.0
    z_hat = 0.0

    T_log = []
    U_log = []

    print("=== MPC TEST: step-change reference ===")
    print("k |  T_k     u_k    Tref_k")
    print("----------------------------------")

    for k in range(sim_len):
        u_k = solve_mpc(
            T, u_prev, z_hat,
            k, T_ref_list,
            a, b,
            N,
            beta, lam, eta, gamma,
            u_min, u_max, du_max, u_peak,
            T_cap
        )

        T = a * T + b * u_k
        Tref_now = T_ref_list[k]

        print(f"{k:2d} | {T:7.2f}  {u_k:6.3f}   {Tref_now:6.1f}")

        T_log.append(T)
        U_log.append(u_k)

        u_prev = u_k

    print("\nFinal T =", T)

    # -------------------------------
    # 시각화
    # -------------------------------
    plot_tracking(T_log, T_ref_list)
