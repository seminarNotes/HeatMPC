# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_preprocess(path, dT_threshold=1.0, u_threshold=1.0):
    df_raw = pd.read_csv(path)

    # 날짜 파싱만 내부 정렬용으로 사용
    df_raw["ts_tmp"] = pd.to_datetime(
        df_raw["일자"].astype(str).str.strip(),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce"
    )

    # 오름차순 정렬
    df = df_raw.sort_values("ts_tmp").reset_index(drop=True)

    # row_index를 index(k)로 사용
    df["i_tmp"] = np.arange(len(df), dtype=int)
    df = df.set_index("i_tmp")

    # 온도=0 → 보간
    T = df["온도1"].values.astype(float)
    if (T == 0).any():
        T[T == 0] = np.nan
        df["온도1"] = pd.Series(T).interpolate().ffill().bfill().values

    # 제어입력 kW = 60*kWh, 0~1 정규화
    df["u_kW"] = 60 * df["kWh"].values.astype(float)
    p_max = df["u_kW"].max()
    df["u"] = (df["u_kW"] / 100.0).clip(0, 1)

    # I / λ 라벨링 (정규화 전 로직 유지)
    df["I"] = 0
    df["lambda"] = np.nan

    dT = df["온도1"].diff().values
    cond1 = (dT >= dT_threshold) & (df["u_kW"].values >= u_threshold)
    i1_idx = np.where(cond1)[0]
    i1_start = i1_idx[0] if len(i1_idx) > 0 else 0

    cond2 = df["온도1"].values >= 910
    i2_idx = np.where(cond2[i1_start:])[0]
    if len(i2_idx) == 0:
        print("[WARN] I2 not found"); return df
    i2_start = i1_start + i2_idx[0]

    cond3 = (df["온도1"].values >= 860) & (df["온도1"].values < 910)
    i3_idx = np.where(cond3[i2_start:])[0]
    if len(i3_idx) == 0:
        print("[WARN] I3 not found"); return df
    i3_start = i2_start + 1 + i3_idx[0]
    i3_end = i2_start + 1 + i3_idx[-1]

    # Connected & Disjoint 할당
    df.loc[i1_start:i2_start-1, "I"] = 1
    df.loc[i2_start:i3_start-1, "I"] = 2
    df.loc[i3_start:i3_end, "I"] = 3

    # λ 값 부여
    for I_val in (1, 2, 3):
        seg = df[df["I"] == I_val].index
        L = len(seg)
        if L > 0:
            df.loc[seg, "lambda"] = np.linspace(0, 1, L) if L > 1 else 0.0

    # ref_T 필드 추가 및 값 채우기
    df["ref_T"] = ""
    # I1, I2 → 930
    seg_930 = df[(df["I"] == 1) | (df["I"] == 2)].index
    df.loc[seg_930, "ref_T"] = 930
    # I3 → 880
    seg_880 = df[df["I"] == 3].index
    df.loc[seg_880, "ref_T"] = 880

    # ts_tmp, row_index 등 임시 컬럼 제거
    if "ts_tmp" in df.columns: df = df.drop(columns=["ts_tmp"])
    if "ts_tmp" in df.index.names: pass
    if "row_index" in df.columns: df = df.drop(columns=["row_index"])
    if "ts_tmp" in df.columns: df = df.drop(columns=["ts_tmp"])
    if "ts_tmp" in df.index.names: pass
    # drop ts_tmp column if exists
    if "ts_tmp" in df.columns: df = df.drop(columns=["ts_tmp"])
    if "ts_tmp" in seg_930: pass  # 안전, 무시됨

    # row number column TS_TMP
    if "ts_tmp" in df.columns: df = df.drop(columns=["ts_tmp"])

    # row number는 index이므로 "ts_tmp"만 제거
    if df_raw is not None:
        if "ts_tmp" in df.columns: pass

    # 내부 정렬용 ts_tmp 제거
    if "ts_tmp" in df.columns:
        df = df.drop(columns=["ts_tmp"])
    if "ts_tmp" in df.index.names:
        df.index.names = [n for n in df.index.names if n != "ts_tmp"]

    return df[["온도1", "u_kW", "u", "I", "lambda", "ref_T"]].copy()


def plot_preprocessed(df):
    if df.empty:
        print("[WARN] empty df"); return

    x = df.index.values
    T_plot = df["온도1"].values
    u_plot = df["u"].values
    I_plot = df["I"].values

    plt.figure(figsize=(10,4))
    plt.plot(x, T_plot, linewidth=1)
    for I_val in (1,2,3):
        idxs = np.where(I_plot == I_val)[0]
        if len(idxs) > 0:
            plt.axvspan(x[idxs[0]], x[idxs[-1]], alpha=0.2)
    plt.title("Temperature with intervals")
    plt.xlabel("k"); plt.ylabel("T")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(10,3))
    plt.plot(x, u_plot, linewidth=1)
    for I_val in (1,2,3):
        idxs = np.where(I_plot == I_val)[0]
        if len(idxs) > 0:
            plt.axvspan(x[idxs[0]], x[idxs[-1]], alpha=0.2)
    plt.title("Input with intervals")
    plt.xlabel("k"); plt.ylabel("u")
    plt.tight_layout(); plt.show()

def save_preprocessed_dataset(df, original_path):
    root = os.path.dirname(original_path)

    # ★ 저장 폴더명 변경
    out_dir = os.path.join(root, "dat_preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    # ★ 입력 파일명 그대로 사용
    file_name = os.path.basename(original_path)
    save_path = os.path.join(out_dir, file_name)

    df.to_csv(save_path, index=False)
    print("[INFO] fields saved:", list(df.columns))
    print("[INFO] saved as:", save_path)
    return save_path



if __name__ == "__main__":
    # 여러 파일을 순회 처리하도록 구성
    data_files = [
        '/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1128.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1130.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1201_1.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1201_2.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1201_3.csv',
        # '/home/junhuiwoo/Desktop/HeatMPC/data/filtered_1128.csv',
    ]

    COLMAP = { "timestamp":"일자", "T":"온도1", "u_kW":"kWh" }

    for f in data_files:
        print("\n[PROCESS]", f)
        df_prep = load_and_preprocess(f)
        print("[INFO] rows after preprocess:", len(df_prep))
        save_path = save_preprocessed_dataset(df_prep, f)
        print("[OK] saved:", save_path)
        plot_preprocessed(df_prep)


