# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------
#  IMPORT: ID, ESO, MPC
# -------------------------------
from step1_dat import load_and_preprocess
from step2_sys import fit_ARX1_ab

from step3_eso import (
    eso_gain,
    eso_reset,
    eso_step,
    eso_get_state,
)

from KNU4_mpc import (
    solve_mpc,
)

# ===========================================================
# 공통 유틸
# ===========================================================
def make_output_dir():
    """
    ./mpc_result/run_YYYYMMDD_HHMMSS 형태의 폴더 생성 후 경로 반환
    """
    base = "./mpc_result"
    os.makedirs(base, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, f"run_{now}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_params(out_dir, params):
    """
    params: dict
      - 값이 dict이면 섹션으로 보고 [SECTION] 헤더를 출력
      - 값이 dict가 아니면 단일 key=value 로 출력
    """
    path = os.path.join(out_dir, "params.txt")
    with open(path, "w", encoding="utf-8") as f:
        for key, val in params.items():
            if isinstance(val, dict):
                f.write(f"[{key}]\n")
                for k2, v2 in val.items():
                    f.write(f"{k2} = {v2}\n")
                f.write("\n")
            else:
                f.write(f"{key} = {val}\n")
    print(f"[INFO] Saved params → {path}")


def save_plot(x, y_list, labels, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(8, 4))
    for y, lb in zip(y_list, labels):
        plt.plot(x, y, label=lb)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot → {save_path}")


# ===========================================================
# (옛) 테스트용 참조 생성 함수 – 지금은 사용 안 함
# ===========================================================
def build_T_test():
    T_ref_list = []
    
    if False:
        K = [0,   250, 430, 440, 500]
        T = [600, 930, 930, 880, 880]
        for k in range(K[-1] + 1):
            if k <= K[1]:
                k0, k1 = K[0], K[1]
                T0, T1 = T[0], T[1]
            elif k <= K[2]:
                k0, k1 = K[1], K[2]
                T0, T1 = T[1], T[2]
            elif k <= K[3]:
                k0, k1 = K[2], K[3]
                T0, T1 = T[2], T[3]
            else:
                k0, k1 = K[3], K[4]
                T0, T1 = T[3], T[4]

            if k1 == k0:
                T_ref = T1
            else:
                tau = (k - k0) / float(k1 - k0)
                T_ref = T0 + (T1 - T0) * tau

            T_ref_list.append(T_ref)

    else:
        K = [0,   350, 351, 500]
        T = [930, 930, 880, 880]

        for k in range(K[-1] + 1):
            if k <= K[1]:
                k0, k1 = K[0], K[1]
                T0, T1 = T[0], T[1]
            elif k <= K[2]:
                k0, k1 = K[1], K[2]
                T0, T1 = T[1], T[2]
            elif k <= K[3]:
                k0, k1 = K[2], K[3]
                T0, T1 = T[2], T[3]
            elif k <= K[4]:
                k0, k1 = K[3], K[4]
                T0, T1 = T[3], T[4]
            else:
                # E → F
                k0, k1 = K[4], K[5]
                T0, T1 = T[4], T[5]

            if k1 == k0:
                T_ref = T1
            else:
                tau = (k - k0) / float(k1 - k0)
                T_ref = T0 + (T1 - T0) * tau

            T_ref_list.append(T_ref)

    return T_ref_list


def build_process_reference(max_steps):
    """
    공정 레시피 기반 참조의 "껍데기"를 생성.
    - 처음에는 전체를 T1_TARGET으로 채워두고,
    - 시뮬레이션 도중 T1 상태기반 hold가 끝나는 시점에
      T2 단계로 넘어갈 때, 해당 시점 이후의 원소를 T2_TARGET으로 덮어쓴다.
    """
    return [T1_TARGET] * max_steps


# ===========================================================
# MODEL / ESO PROPERTY CHECK
# ===========================================================
def analyze_system_properties(a_hat, b_hat, L):
    """
    - ARX(1) stability (|a_hat| < 1 ?)
    - Extended (A,C) observability
    - ESO error dynamics stability (eig(A - L C))
    """
    print("\n[INFO] ==== Model / ESO Analysis ====")
    print(f"[INFO] Identified ARX(1): T_(k+1) = {a_hat:.6f} * T_k + {b_hat:.6f} * u_k + z_k")

    # 1) ARX(1) stability
    abs_a = abs(a_hat)
    print(f"[INFO] |a_hat| = {abs_a:.6f}  -> "
          f"{'STABLE (|a|<1)' if abs_a < 1.0 else 'UNSTABLE (|a|>=1)'}")

    # 2) Extended system (A,C) observability
    A = np.array([[a_hat, 1.0],
                  [0.0,   1.0]])
    C = np.array([[1.0, 0.0]])   # y = [1 0] x = T

    O = np.vstack((C, C @ A))    # Observability matrix [C; C A]
    rank_O = np.linalg.matrix_rank(O)
    print(f"[INFO] Observability matrix O =\n{O}")
    print(f"[INFO] rank(O) = {rank_O} -> "
          f"{'OBSERVABLE (rank=2)' if rank_O == 2 else 'NOT full observable'}")

    # 3) ESO error dynamics A_cl = A - L C
    L = np.asarray(L).reshape(2, 1)    # ensure 2x1
    A_cl = A - L @ C                   # 2x2
    eigvals = np.linalg.eigvals(A_cl)
    max_abs_eig = max(abs(eigvals))
    print(f"[INFO] ESO error dynamics A-LC =\n{A_cl}")
    print(f"[INFO] eig(A-LC) = {eigvals}")
    print(f"[INFO] max |eig| = {max_abs_eig:.6f} -> "
          f"{'STABLE (all inside unit circle)' if max_abs_eig < 1.0 else 'UNSTABLE'}")
    print("[INFO] =================================\n")


# ===========================================================
# CLOSED-LOOP SIM FOR ONE INITIAL CONDITION (while-loop 버전)
# ===========================================================
def run_closed_loop_sim(
    df,
    a_hat,
    b_hat,
    L,
    N_horizon,
    beta,
    lam,
    eta,
    gamma,
    u_min,
    u_max,
    du_max,
    u_peak,
    T_cap,
    T_init,
    base_out_dir,
    label=None,
):
    """
    - while-loop 기반 동적 시뮬레이션
    - 공정 종료 조건:
        1) |T - T1_TARGET| <= TEMP_TOL 을 연속 HOLD_T1_HOURS 동안 만족
        2) 이후 ref를 T2_TARGET으로 전환
        3) |T - T2_TARGET| <= TEMP_TOL 을 연속 HOLD_T2_HOURS 동안 만족
      → 위 조건을 만족하는 시점에서 시뮬레이션 종료

    CSV 컬럼:
        k      : time step
        T_ref  : reference temperature
        T      : plant temperature
        u      : control input (normalized)
        flag   : 'mpc' / 'fallback' 구분 (현재는 모두 'mpc')
        z_hat  : ESO disturbance estimate
    """

    # 안전 상한: "최대 시뮬레이션 시간" (heat-up/transition 실패 방지용)
    # 예: T1_hold + T2_hold + margin_hours 만큼
    max_sim_hours = HOLD_T1_HOURS + HOLD_T2_HOURS + SIM_MARGIN_HOURS
    MAX_STEPS = int(max_sim_hours * STEPS_PER_HOUR)

    # 공정 레시피 기반 참조의 껍데기 (처음엔 전구간 T1으로 채움)
    T_ref_list = build_process_reference(MAX_STEPS)

    # ESO 초기 상태 (T_init, z=0)
    x_hat = eso_reset(T_init, 0.0)

    # 초기 플랜트 상태 및 제어 입력
    T_true = T_init
    u_prev = float(df["u"].iloc[0])  # 초기 입력은 데이터에서 1개 가져와 사용

    # 로그 버퍼 (k=0에서의 상태)
    T_log = [T_true]
    u_log = [u_prev]
    z_log = [0.0]
    flag_log = ["init"]  # k=0은 MPC가 아니라 초기값이므로 'init'
    Tref_log = [T1_TARGET]  # k=0에서의 ref (초기에는 T1을 목표로 한다고 가정)

    desc = f"[SIM T_init={T_init}]" if label is None else f"[SIM {label}]"
    print(f"[INFO] Start simulation {desc}")

    # 상태 기반 hold 카운터 및 metric 변수
    n_hold_T1 = int(HOLD_T1_HOURS * STEPS_PER_HOUR)
    n_hold_T2 = int(HOLD_T2_HOURS * STEPS_PER_HOUR)

    t1_within_count = 0
    t2_within_count = 0

    phase = "T1"  # 현재 어떤 set-point를 추종 중인지 ("T1" → "T2")
    process_done = False

    k_T1_reach = None
    k_T1_hold_start = None
    k_T1_hold_end = None
    k_T2_switch_ref = None
    k_T2_reach = None
    k_T2_hold_start = None
    k_T2_hold_end = None
    k_process_end = None

    # 시뮬레이션 루프 (k는 "현재 step index")
    k = 1
    while k < MAX_STEPS and (not process_done):

        # --------------------------------------------------
        # 1) 현재 phase에 따른 set-point 선택
        # --------------------------------------------------
        if phase == "T1":
            T_ref_now = T1_TARGET
        else:
            T_ref_now = T2_TARGET

        # 참조 리스트에 현재 step의 ref 기록
        T_ref_list[k] = T_ref_now

        # --------------------------------------------------
        # 2) ESO update (이전 step의 T_true, u_prev 기반)
        # --------------------------------------------------
        x_hat = eso_step(T_true, u_prev, x_hat, a_hat, b_hat, L)
        T_hat, z_hat = eso_get_state(x_hat)

        # --------------------------------------------------
        # 3) MPC solve
        # --------------------------------------------------
        sol = solve_mpc(
            T_hat, u_prev, z_hat,
            k, T_ref_list,
            a_hat, b_hat,
            N_horizon,
            beta, lam, eta, gamma,
            u_min, u_max, du_max, u_peak,
            T_cap,
        )

        # solve_mpc 가 u만 반환하는 경우 / (u, flag) 튜플 반환하는 경우 모두 지원
        if isinstance(sol, tuple):
            u_k, flag_k = sol
        else:
            u_k = sol
            flag_k = "mpc"

        # --------------------------------------------------
        # 4) Plant update (ARX(1) + ESO disturbance)
        # --------------------------------------------------
        T_next = a_hat * T_true + b_hat * u_k + z_hat

        T_est = T_hat
        T_err_ref = T_est - T_ref_now

        print(f"[{k:04d}] "
              f"T_ref : {T_ref_now:7.2f}, "
              f"T_est : {T_est:7.2f}, "
              f"T_err : {T_err_ref:7.2f}, "
              f"u_mpc : {u_k:6.3f}, "
              f"z_hat : {z_hat:7.3f}")

        # ---- EARLY BREAK CONDITION (이상 발산 방지용) ----
        if abs(T_err_ref) > 500:
            print(f"[WARN] |T_err_ref| > 500 at k={k}, early stop.")
            k_process_end = k
            T_true = T_next
            T_log.append(T_true)
            u_log.append(u_k)
            z_log.append(z_hat)
            flag_log.append(flag_k)
            Tref_log.append(T_ref_now)
            break

        # --------------------------------------------------
        # 5) 로그 업데이트
        # --------------------------------------------------
        T_true = T_next
        u_prev = u_k

        T_log.append(T_true)
        u_log.append(u_k)
        z_log.append(z_hat)
        flag_log.append(flag_k)
        Tref_log.append(T_ref_now)

        # --------------------------------------------------
        # 6) 상태 기반 hold 조건 업데이트
        # --------------------------------------------------
        if phase == "T1":
            # T1 band check
            if abs(T_true - T1_TARGET) <= TEMP_TOL:
                if k_T1_reach is None:
                    k_T1_reach = k  # 최초 band 진입 시점
                t1_within_count += 1
            else:
                t1_within_count = 0  # 연속 구간 깨짐

            # T1 hold 완료?
            if (k_T1_hold_end is None) and (t1_within_count >= n_hold_T1):
                k_T1_hold_end = k
                k_T1_hold_start = k_T1_hold_end - n_hold_T1 + 1
                # 다음 step부터 T2를 목표로
                k_T2_switch_ref = k + 1
                phase = "T2"
                # 참조 리스트를 전구간 T2로 덮어써도 horizon 관점에서는 OK
                for idx in range(k_T2_switch_ref, MAX_STEPS):
                    T_ref_list[idx] = T2_TARGET
                print(f"[INFO] T1 hold completed at k={k_T1_hold_end}, "
                      f"switch ref to T2 at k={k_T2_switch_ref}")

        elif phase == "T2":
            # T2 band check
            if abs(T_true - T2_TARGET) <= TEMP_TOL:
                if k_T2_reach is None:
                    k_T2_reach = k  # 최초 band 진입 시점
                t2_within_count += 1
            else:
                t2_within_count = 0  # 연속 구간 깨짐

            # T2 hold 완료 → 공정 종료
            if (k_T2_hold_end is None) and (t2_within_count >= n_hold_T2):
                k_T2_hold_end = k
                k_T2_hold_start = k_T2_hold_end - n_hold_T2 + 1
                k_process_end = k_T2_hold_end
                process_done = True
                print(f"[INFO] T2 hold completed at k={k_T2_hold_end}, process finished.")
                break

        # step 증가
        k += 1

        if k == 3 :
            print (1)
            break

    # while 루프가 자연 종료되었는데 process_done이 False이면,
    # max step에 도달했거나 early break였다고 판단하고, 마지막 index를 종료시점으로 둔다.
    if k_process_end is None:
        k_process_end = len(T_log) - 1

    sim_len = len(T_log)
    k_arr = np.arange(sim_len)

    # ---------------------------------------------
    # 성능지표 정리
    # ---------------------------------------------
    energy_sum = float(np.sum(u_log[:k_process_end + 1]))

    # suffix 정의 (파일 이름용)
    if label is None:
        suffix = ""
    else:
        suffix = f"_{label}"

    print("[METRIC] ---- Process metrics ----")
    print(f"[METRIC] k_T1_reach          = {k_T1_reach}")
    print(f"[METRIC] k_T1_hold_start     = {k_T1_hold_start}")
    print(f"[METRIC] k_T1_hold_end       = {k_T1_hold_end}")
    print(f"[METRIC] k_T2_switch_ref     = {k_T2_switch_ref}")
    print(f"[METRIC] k_T2_reach          = {k_T2_reach}")
    print(f"[METRIC] k_T2_hold_start     = {k_T2_hold_start}")
    print(f"[METRIC] k_T2_hold_end       = {k_T2_hold_end}")
    print(f"[METRIC] k_process_end       = {k_process_end}")
    if k_T1_reach is not None:
        print(f"[METRIC] t_T1_reach (hours)      = {k_T1_reach / STEPS_PER_HOUR:.3f}")
    if k_T2_reach is not None:
        print(f"[METRIC] t_T2_reach (hours)      = {k_T2_reach / STEPS_PER_HOUR:.3f}")
    print(f"[METRIC] t_process_end (hours)   = {k_process_end / STEPS_PER_HOUR:.3f}")
    print(f"[METRIC] Energy sum (∑u)         = {energy_sum:.4f}")
    print("[METRIC] ------------------------")

    # 성능지표 CSV 저장
    metrics = {
        "k_T1_reach": k_T1_reach,
        "k_T1_hold_start": k_T1_hold_start,
        "k_T1_hold_end": k_T1_hold_end,
        "k_T2_switch_ref": k_T2_switch_ref,
        "k_T2_reach": k_T2_reach,
        "k_T2_hold_start": k_T2_hold_start,
        "k_T2_hold_end": k_T2_hold_end,
        "k_process_end": k_process_end,
        "t_T1_reach_hours": (k_T1_reach / STEPS_PER_HOUR) if k_T1_reach is not None else None,
        "t_T2_reach_hours": (k_T2_reach / STEPS_PER_HOUR) if k_T2_reach is not None else None,
        "t_process_end_hours": k_process_end / STEPS_PER_HOUR,
        "energy_sum_u": energy_sum,
    }
    metrics_df = pd.DataFrame([metrics])
    os.makedirs(base_out_dir, exist_ok=True)
    safe_suffix = Path(suffix).stem
    metrics_csv_path = os.path.join(base_out_dir, f"metrics_{safe_suffix}.csv")
    metrics_csv_path = os.path.join(base_out_dir, f"metrics{suffix}.csv")
    print (base_out_dir)
    print (metrics_csv_path)
    print (111111)
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[INFO] Saved metrics → {metrics_csv_path}")

    # ---------------------------------------------
    # 결과 CSV 저장
    # ---------------------------------------------
    result_df = pd.DataFrame({
        "k": k_arr,
        "T_ref": Tref_log,
        "T": T_log,
        "u": u_log,
        "flag": flag_log,
        "z_hat": z_log,
    })
    result_csv_path = os.path.join(base_out_dir, f"sim{suffix}.csv")
    print (result_csv_path)
    result_df.to_csv(result_csv_path, index=False)
    print(f"[INFO] Saved sim result → {result_csv_path}")

    # ---------------------------------------------
    # PLOT: Temperature + reference (+ 수직 점선)
    # ---------------------------------------------
    tracking_path = os.path.join(base_out_dir, f"tracking{suffix}.png")
    plt.figure(figsize=(8, 4))
    plt.plot(k_arr, T_log, label="T")
    plt.plot(k_arr, Tref_log, label="T_ref")

    # T1 도달 시점
    if k_T1_reach is not None:
        plt.axvline(
            k_T1_reach,
            linestyle="--",
            color="C2",
            label=f"T1 = {T1_TARGET:.0f} reach at k = {k_T1_reach}",
        )

    # T2 도달 시점
    if k_T2_reach is not None:
        plt.axvline(
            k_T2_reach,
            linestyle="--",
            color="C3",
            label=f"T2 = {T2_TARGET:.0f} reach at k = {k_T2_reach}",
        )

    # 공정 종료 시점
    if k_process_end is not None:
        plt.axvline(
            k_process_end,
            linestyle="--",
            color="C4",
            label=f"Process end at k = {k_process_end}",
        )

    plt.title(f"Temperature Tracking{suffix}")
    plt.xlabel("k")
    plt.ylabel("Temperature")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(tracking_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot → {tracking_path}")

    # ---------------------------------------------
    # 나머지 플롯은 기존 save_plot 그대로 사용
    # ---------------------------------------------
    # PLOT: Tracking error (T - T_ref)
    err = np.array(T_log) - np.array(Tref_log)
    save_plot(
        k_arr,
        [err],
        ["T - T_ref"],
        f"Tracking Error{suffix}",
        "k",
        "Error",
        os.path.join(base_out_dir, f"error{suffix}.png"),
    )

    # PLOT: Control input
    save_plot(
        k_arr,
        [u_log],
        ["u"],
        f"Control Input{suffix}",
        "k",
        "u",
        os.path.join(base_out_dir, f"control{suffix}.png"),
    )

    # PLOT: Disturbance estimate
    save_plot(
        k_arr,
        [z_log],
        ["z_hat"],
        f"ESO disturbance estimate{suffix}",
        "k",
        "z_hat",
        os.path.join(base_out_dir, f"disturbance{suffix}.png"),
    )


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__" and True:
    # ===========================================================
    # 공정 시나리오 설정 (T1=930, T2=880)
    # ===========================================================
    T1_TARGET = 930.0
    T2_TARGET = 880.0
    TEMP_TOL  = 5.0          # |T - T_ref| <= 5 허용 오차
    HOLD_T1_HOURS = 3.0      # T1 유지 시간 (시간 단위) - 필요시 33.0으로 변경
    HOLD_T2_HOURS = 1.0      # T2 유지 시간 (시간 단위)
    STEPS_PER_HOUR = 60      # RESAMPLE_MIN=1분 가정 → 1시간 = 60스텝
    SIM_MARGIN_HOURS = 10.0  # heat-up/transition 실패 대비 여유 시간

    # -------------------------------
    # CONFIG (실험 공통 설정)
    # -------------------------------
    # DATA_PATH_LIST = [
    #     r"D:\5.Project\KNU\data\filtered_1128.csv",
    #     r"D:\5.Project\KNU\data\filtered_1130.csv",
    #     r"D:\5.Project\KNU\data\filtered_1201_1.csv",
    #     r"D:\5.Project\KNU\data\filtered_1201_2.csv",
    #     r"D:\5.Project\KNU\data\filtered_1201_3.csv",
    # ]

    DATA_PATH_LIST = [
        "/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1128.csv",
        "/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1130.csv",
        "/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1201_1.csv",
        "/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1201_2.csv",
        "/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1201_3.csv",
    ]

    COLMAP = {"timestamp": "일자", "T": "온도1", "u_kW": "kWh"}

    RESAMPLE_MIN = 1
    U_UNIT = "kWh"
    P_MAX_KW = 100.0
    COLD_LOAD_THRESHOLD = -1.5
    COLD_LOAD_WINDOW = 3

    # ESO poles
    ESO_P1, ESO_P2 = 0.2, 0.3

    # MPC config (soft constraints)
    N_horizon = 30
    beta = 2.0
    lam  = 100.0
    eta  = 0.01
    gamma = 0.1 

    # MPC config (hard constraints)
    u_min = 0.0
    u_max = 1.0
    du_max = 0.5  # 0.1
    u_peak = 1.0
    T_cap = 1200.0


    # -------------------------------
    # 각 DATA_PATH 별로 독립적인 run_* 디렉토리 생성 및 실험 수행
    # -------------------------------
    

    # for DATA_PATH in RAW_DATA_PATH_LIST : 
    DATA_PATH = DATA_PATH_LIST[0]
    df_prep = load_and_preprocess(DATA_PATH)
    # a, b1, b2, b3, rmse = fit_global_ab(df_prep)
    # print (df_prep)
    print () # 데이터의 첫 시점 온도
    a_hat, b_hat, *_ = fit_ARX1_ab(df_prep)
    print (1)
 


    for DATA_PATH in DATA_PATH_LIST:

        print("\n" + "=" * 80)
        print(f"[INFO] NEW RUN for DATA_PATH = {DATA_PATH}")
        print("=" * 80)

        # -------------------------------
        # OUTPUT DIR (이 데이터셋에 대한 전용 run 디렉토리)
        # -------------------------------
        base_out_dir = make_output_dir()
        print(f"[INFO] Output base directory → {base_out_dir}")

        # -------------------------------
        # STEP 1: LOAD DATA
        # -------------------------------
        df_prep = load_and_preprocess(DATA_PATH)
        T_init = df_prep.iloc[0]['온도']

        print(f"[INFO] Saved processed data → {DATA_PATH}")

        # -------------------------------
        # STEP 2: SYSTEM ID
        # -------------------------------
        a_hat, b_hat, _, _, rmse = fit_ARX1_ab(df_prep)
        print(f"[INFO] ID result: a_hat={a_hat:.6f}, b_hat={b_hat:.6f}, rmse={rmse:.6f}")

        # ESO 이득 계산
        L, _, _ = eso_gain(a_hat, ESO_P1, ESO_P2)
        print(f"[INFO] ESO gain L = {L}")

        # 모델/ESO 특성 분석 로그
        analyze_system_properties(a_hat, b_hat, L)

        # -------------------------------
        # PARAM SUMMARY 저장
        # -------------------------------
        params = {
            "DATA": {
                "DATA_PATH": DATA_PATH,
                "DATA_NAME": os.path.basename(DATA_PATH),
                "RESAMPLE_MIN": RESAMPLE_MIN,
                "U_UNIT": U_UNIT,
                "P_MAX_KW": P_MAX_KW,
                "COLD_LOAD_THRESHOLD": COLD_LOAD_THRESHOLD,
                "COLD_LOAD_WINDOW": COLD_LOAD_WINDOW,
            },
            "ID": {
                "a_hat": a_hat,
                "b_hat": b_hat,
                "rmse": rmse,
            },
            "ESO": {
                "ESO_P1": ESO_P1,
                "ESO_P2": ESO_P2,
                "L_0": float(L[0]),
                "L_1": float(L[1]),
            },
            "MPC": {
                "N_horizon": N_horizon,
                "beta": beta,
                "lam": lam,
                "eta": eta,
                "gamma": gamma,
                "u_min": u_min,
                "u_max": u_max,
                "du_max": du_max,
                "u_peak": u_peak,
                "T_cap": T_cap,
            },
            "SIM": {
                "T_init": T_init,
                "T1_TARGET": T1_TARGET,
                "T2_TARGET": T2_TARGET,
                "TEMP_TOL": TEMP_TOL,
                "HOLD_T1_HOURS": HOLD_T1_HOURS,
                "HOLD_T2_HOURS": HOLD_T2_HOURS,
                "STEPS_PER_HOUR": STEPS_PER_HOUR,
                "SIM_MARGIN_HOURS": SIM_MARGIN_HOURS,
            },
        }
        save_params(base_out_dir, params)

        # -------------------------------
        # 여러 초기 조건에 대해 시뮬레이션 수행
        # -------------------------------

        label = f"csvfile_{str(DATA_PATH.split('/')[-1][:-4])}" #csv 파일 이름으로 변경해야함.
        # print (label)
        
        run_closed_loop_sim(
            df=df_prep,
            a_hat=a_hat,
            b_hat=b_hat,
            L=L,
            N_horizon=N_horizon,
            beta=beta,
            lam=lam,
            eta=eta,
            gamma=gamma,
            u_min=u_min,
            u_max=u_max,
            du_max=du_max,
            u_peak=u_peak,
            T_cap=T_cap,
            T_init=T_init,
            base_out_dir=base_out_dir,
            label=label,
        )

        print(f"[INFO] All experiments finished for {DATA_PATH}\n")

