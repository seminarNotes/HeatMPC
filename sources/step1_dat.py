# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_preprocess(path, dT_threshold=1.0, u_threshold=1.0):
    """
    새 형식 CSV 예:
    생산시간,온도,전력량(1분),누적전력량
    2025-11-28 12:40:02,838,1.9,6984.41
    ...

    - 생산시간: 단순히 row index가 k (1분 샘플링 가정)
    - 온도: 온도 컬럼 사용 (0이면 보간)
    - 전력량(1분): 분당 에너지[kWh]로 가정하여
        u_kW = 60 * 전력량(1분)
        u = (u_kW / 100).clip(0, 1)
    - 나머지 I, lambda, ref_T 로직은 기존과 동일
    """
    df_raw = pd.read_csv(path)

    # 1) 정렬/파싱 없이, 읽힌 순서대로 k = 0,1,2,... 부여
    df = df_raw.reset_index(drop=True)
    df["i_tmp"] = np.arange(len(df), dtype=int)
    df = df.set_index("i_tmp")

    # 2) 온도 컬럼 이름 확인 (새 데이터는 "온도")
    if "온도" not in df.columns:
        raise ValueError("입력 파일에 '온도' 컬럼이 없습니다.")
    T_col = "온도"

    # 온도 = 0 → 보간 (기존 로직 유지)
    T = df[T_col].values.astype(float)
    if (T == 0).any():
        T[T == 0] = np.nan
        df[T_col] = pd.Series(T).interpolate().ffill().bfill().values

    # 3) 제어입력: 전력량(1분) → kW → 0~1 정규화
    if "전력량(1분)" not in df.columns:
        raise ValueError("입력 파일에 '전력량(1분)' 컬럼이 없습니다.")

    # 전력량(1분): [kWh/분]으로 보고, kW = 60 * kWh/분
    df["u_kW"] = 60.0 * df["전력량(1분)"].values.astype(float)

    # 최대 전력은 참고용 (필요하면 사용)
    p_max = df["u_kW"].max()

    # 0~1 정규화 (기존과 동일하게 /100 후 clip)
    df["u"] = (df["u_kW"] / 100.0).clip(0.0, 1.0)

    # 4) I / lambda 라벨링 (기존 로직 동일, 단 T_col 사용)
    df["I"] = 0
    df["lambda"] = np.nan

    dT = df[T_col].diff().values

    # I1 시작: dT >= dT_threshold 이고, u_kW >= u_threshold
    cond1 = (dT >= dT_threshold) & (df["u_kW"].values >= u_threshold)
    i1_idx = np.where(cond1)[0]
    i1_start = i1_idx[0] if len(i1_idx) > 0 else 0

    # I2 시작: 온도 >= 910
    cond2 = df[T_col].values >= 910.0
    i2_idx = np.where(cond2[i1_start:])[0]
    if len(i2_idx) == 0:
        print("[WARN] I2 not found")
        return df[[T_col, "u_kW", "u", "I", "lambda"]].copy()
    i2_start = i1_start + i2_idx[0]

    # I3: 860 <= 온도 < 910
    cond3 = (df[T_col].values >= 860.0) & (df[T_col].values < 910.0)
    i3_idx = np.where(cond3[i2_start:])[0]
    if len(i3_idx) == 0:
        print("[WARN] I3 not found")
        return df[[T_col, "u_kW", "u", "I", "lambda"]].copy()
    i3_start = i2_start + 1 + i3_idx[0]
    i3_end   = i2_start + 1 + i3_idx[-1]

    # 구간 라벨 I = 1,2,3
    df.loc[i1_start:i2_start-1, "I"] = 1
    df.loc[i2_start:i3_start-1, "I"] = 2
    df.loc[i3_start:i3_end,     "I"] = 3

    # 5) λ 값 부여 (각 구간마다 0→1 선형 스케일)
    for I_val in (1, 2, 3):
        seg = df[df["I"] == I_val].index
        L = len(seg)
        if L > 0:
            df.loc[seg, "lambda"] = np.linspace(0.0, 1.0, L) if L > 1 else 0.0

    # 6) ref_T 필드 추가 (I1,I2: 930 / I3: 880)
    df["ref_T"] = np.nan
    seg_930 = df[(df["I"] == 1) | (df["I"] == 2)].index
    seg_880 = df[df["I"] == 3].index
    df.loc[seg_930, "ref_T"] = 930.0
    df.loc[seg_880, "ref_T"] = 880.0

    # 최종 반환 (온도 컬럼 이름은 그대로 유지)
    return df[[T_col, "u_kW", "u", "I", "lambda", "ref_T"]].copy()


def plot_preprocessed(df):
    if df is None or df.empty:
        print("[WARN] empty df")
        return

    x = df.index.values

    # 온도 컬럼 이름 자동 탐색
    if "온도" in df.columns:
        T_col = "온도"
    elif "온도1" in df.columns:
        T_col = "온도1"
    else:
        raise ValueError("df에 온도 컬럼(온도 / 온도1)이 없습니다.")

    T_plot = df[T_col].values
    u_plot = df["u"].values
    I_plot = df["I"].values

    plt.figure(figsize=(10, 4))
    plt.plot(x, T_plot, linewidth=1)
    for I_val in (1, 2, 3):
        idxs = np.where(I_plot == I_val)[0]
        if len(idxs) > 0:
            plt.axvspan(x[idxs[0]], x[idxs[-1]], alpha=0.2)
    plt.title("Temperature with intervals")
    plt.xlabel("k")
    plt.ylabel("T")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 3))
    plt.plot(x, u_plot, linewidth=1)
    for I_val in (1, 2, 3):
        idxs = np.where(I_plot == I_val)[0]
        if len(idxs) > 0:
            plt.axvspan(x[idxs[0]], x[idxs[-1]], alpha=0.2)
    plt.title("Input with intervals")
    plt.xlabel("k")
    plt.ylabel("u")
    plt.tight_layout()
    plt.show()


def save_preprocessed_dataset(df, original_path):
    root = os.path.dirname(original_path)
    out_dir = os.path.join(root, "dat_preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    file_name = os.path.basename(original_path)
    save_path = os.path.join(out_dir, file_name)

    df.to_csv(save_path, index=False)
    print("[INFO] fields saved:", list(df.columns))
    print("[INFO] saved as:", save_path)
    return save_path


if __name__ == "__main__":
    data_files = [
        "/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1128.csv",
        "/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1130.csv",
        "/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1201_1.csv",
        "/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1201_2.csv",
        "/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1201_3.csv",
    ]
    data_files = ['/home/junhuiwoo/Desktop/HeatMPC/data/targeted_1213.csv']

    for f in data_files:
        print("\n[PROCESS]", f)
        df_prep = load_and_preprocess(f)
        print("[INFO] rows after preprocess:", len(df_prep))
        save_path = save_preprocessed_dataset(df_prep, f)
        print("[OK] saved:", save_path)
        plot_preprocessed(df_prep)
