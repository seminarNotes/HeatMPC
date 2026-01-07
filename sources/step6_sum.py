# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# -------------------------------------------------------
# 비용함수 파라미터 (MPC와 동일)
# -------------------------------------------------------
beta = 2.0
lam  = 100.0
eta  = 0.01
gamma = 0.1   # 여기서는 cost 계산에 직접 사용되진 않음 (피크 패널티X)

# -------------------------------------------------------
# PID(Data) 측 성능 계산 : 전체 구간, ref_T 컬럼을 목표온도로 사용
#   CSV: 온도,u_kW,u,I,lambda,ref_T
# -------------------------------------------------------
def compute_metrics_data_window(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # 필요한 컬럼: 온도, ref_T, u
    required_cols = ["온도", "ref_T", "u"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Data CSV에 '{col}' 컬럼이 없습니다.")
    # 숫자로 변환
    df["온도"]  = pd.to_numeric(df["온도"],  errors="coerce")
    df["ref_T"] = pd.to_numeric(df["ref_T"], errors="coerce")
    df["u"]     = pd.to_numeric(df["u"],    errors="coerce")

    # NaN 제거 (참조온도 없는 구간은 성능지표에서 제외)
    df = df.dropna(subset=["온도", "ref_T", "u"])

    if df.empty:
        return {"N": 0, "sum_u": np.nan, "sum_e2": np.nan, "cost": np.nan}

    # 전체 구간 사용
    T_all     = df["온도"].to_numpy()
    T_ref_all = df["ref_T"].to_numpy()
    u_all     = df["u"].to_numpy()

    N = len(T_all)
    if N == 0:
        return {"N": 0, "sum_u": np.nan, "sum_e2": np.nan, "cost": np.nan}

    # e_k = T_k - T_ref_k
    e = T_all - T_ref_all

    # 집계 지표
    sum_u  = float(np.sum(u_all))
    sum_e2 = float(np.sum(e**2))

    # Δu_k = u_k - u_{k-1}, Δu_0 = 0
    du = np.diff(u_all, prepend=u_all[0])
    du[0] = 0.0

    # stage cost = beta * e^2 + lam * du^2 + eta * u^2
    stage_cost = beta * (e**2) + lam * (du**2) + eta * (u_all**2)
    J = float(np.sum(stage_cost))

    return {
        "N": int(N),
        "sum_u": sum_u,
        "sum_e2": sum_e2,
        "cost": J,
    }

# -------------------------------------------------------
# MPC 측 성능 계산 : 전체 구간, T_ref 컬럼을 목표온도로 사용
#   CSV: k,T_ref,T,u,flag,z_hat
# -------------------------------------------------------
def compute_metrics_mpc_window(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # 필요한 컬럼: T_ref, T, u
    required_cols = ["T_ref", "T", "u"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"MPC CSV에 '{c}' 컬럼이 없습니다.")

    # 숫자로 변환
    df["T_ref"] = pd.to_numeric(df["T_ref"], errors="coerce")
    df["T"]     = pd.to_numeric(df["T"],     errors="coerce")
    df["u"]     = pd.to_numeric(df["u"],     errors="coerce")

    # NaN 제거
    df = df.dropna(subset=["T_ref", "T", "u"])

    if df.empty:
        return {"N": 0, "sum_u": np.nan, "sum_e2": np.nan, "cost": np.nan}

    # 전체 구간 사용
    T_ref_all = df["T_ref"].to_numpy()
    T_all     = df["T"].to_numpy()
    u_all     = df["u"].to_numpy()

    N = len(T_all)
    if N == 0:
        return {"N": 0, "sum_u": np.nan, "sum_e2": np.nan, "cost": np.nan}

    # e_k = T_k - T_ref_k
    e = T_all - T_ref_all

    # Δu_k = u_k - u_{k-1}, Δu_0 = 0
    du = np.diff(u_all, prepend=u_all[0])
    du[0] = 0.0

    # 집계 지표
    sum_u  = float(np.sum(u_all))
    sum_e2 = float(np.sum(e**2))

    # stage cost: beta * e^2 + lam * du^2 + eta * u^2
    stage_cost = beta * (e**2) + lam * (du**2) + eta * (u_all**2)
    J = float(np.sum(stage_cost))

    return {
        "N": int(N),
        "sum_u": sum_u,
        "sum_e2": sum_e2,
        "cost": J,
    }

# -------------------------------------------------------
# MPC 결과 파일 찾기: sim_csvfile_{log_id}.csv 검색
# -------------------------------------------------------
def find_mpc_sim_csv(base_mpc_dir: str, log_id: str):
    """
    base_mpc_dir 아래에서 이름이 sim_csvfile_{log_id}.csv 인 파일을 찾아
    첫 번째로 발견되는 경로를 반환. 찾지 못하면 None.
    """
    target_name = f"sim_csvfile_{log_id}.csv"
    for root, dirs, files in os.walk(base_mpc_dir):
        if target_name in files:
            return os.path.join(root, target_name)
    return None

# -------------------------------------------------------
# 메인 루틴
# -------------------------------------------------------
def main():
    # 스크립트를 /home/junhuiwoo/Desktop/HeatMPC 에서 실행한다고 가정
    base_dat = "./data/dat_preprocessed"
    base_mpc = "./mpc_result"

    run_ids = [
        "filtered_1128",
        "filtered_1130",
        "filtered_1201_1",
        "filtered_1201_2",
        "filtered_1201_3",
    ]

    # 출력 헤더
    header = (
        f"{'Run ID':>15} | "
        f"{'N_data':>7} {'N_mpc':>7} {'↓rate(%)':>9} | "
        f"{'Σu_data':>12} {'Σu_mpc':>12} {'↓rate(%)':>14}| "
        f"{'√Σe2_data':>14} {'√Σe2_mpc':>14} {'↓rate(%)':>16}| "
        f"{'J_data':>14} {'J_mpc':>14} {'↓rate(%)':>16}"
    )
    print(header)
    print("-" * len(header))

    for run_id in run_ids:
        dat_path = os.path.join(base_dat, f"{run_id}.csv")
        mpc_path = find_mpc_sim_csv(base_mpc, run_id)

        # 데이터 CSV 읽기
        try:
            df_dat = pd.read_csv(dat_path, encoding="utf-8-sig")
        except Exception as e:
            print(f"{run_id:>15} | Data CSV 읽기 오류: {e}")
            continue

        # MPC CSV 경로 확인
        if mpc_path is None:
            print(f"{run_id:>15} | MPC CSV 파일(sim_csvfile_{run_id}.csv)을 찾지 못했습니다.")
            continue

        # MPC CSV 읽기
        try:
            df_mpc = pd.read_csv(mpc_path, encoding="utf-8-sig")
        except Exception as e:
            print(f"{run_id:>15} | MPC CSV 읽기 오류: {e}")
            continue

        # 성능 지표 계산
        m_dat = compute_metrics_data_window(df_dat)
        m_mpc = compute_metrics_mpc_window(df_mpc)

        # 감소율(%): (데이터 - MPC) / 기준 * 100
        # N 은 MPC 기준 (샘플 수 감소율, +면 데이터가 더 길다는 의미)
        n_rate = (m_dat["N"] - m_mpc["N"]) / max(m_mpc["N"], 1) * 100.0

        # Σu, √Σe2, J 는 데이터 기준 (값이 클수록 나쁘다고 보고 감소율 계산)
        sum_u_dat   = m_dat["sum_u"]
        sum_u_mpc   = m_mpc["sum_u"]
        sqrt_e2_dat = np.sqrt(m_dat["sum_e2"]) if not np.isnan(m_dat["sum_e2"]) else np.nan
        sqrt_e2_mpc = np.sqrt(m_mpc["sum_e2"]) if not np.isnan(m_mpc["sum_e2"]) else np.nan
        J_dat       = m_dat["cost"]
        J_mpc       = m_mpc["cost"]

        sum_u_rate = (sum_u_dat - sum_u_mpc) / max(sum_u_dat, 1e-9) * 100.0 if not np.isnan(sum_u_dat) else np.nan
        sqrt_e2_rate = (
            (sqrt_e2_dat - sqrt_e2_mpc) / max(sqrt_e2_dat, 1e-9) * 100.0
            if not np.isnan(sqrt_e2_dat)
            else np.nan
        )
        J_rate = (J_dat - J_mpc) / max(J_dat, 1e-9) * 100.0 if not np.isnan(J_dat) else np.nan

        print(
            f"{run_id:>15} | "
            f"{m_dat['N']:7d} {m_mpc['N']:7d} {n_rate:9.3f} | "
            f"{sum_u_dat:12.4f} {sum_u_mpc:12.4f} {sum_u_rate:14.3f} | "
            f"{sqrt_e2_dat:14.4f} {sqrt_e2_mpc:14.4f} {sqrt_e2_rate:16.3f} | "
            f"{J_dat:14.4f} {J_mpc:14.4f} {J_rate:16.3f}"
        )

if __name__ == "__main__":
    main()
