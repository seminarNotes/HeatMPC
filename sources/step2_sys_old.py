import numpy as np
import pandas as pd

from step1_dat_old import load_and_preprocess

def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["I"].isin([1, 2, 3])].copy()
    df["T_next"] = df["온도1"].shift(-1)
    df = df.dropna(subset=["T_next"])
    return df

def prepare_dataset(df) :
    df = df[df["I"].isin([1, 2, 3])].copy()
    df["T_next"] = df["온도1"].shift(-1)
    df = df.dropna(subset=["T_next"])
    return df


def build_design_matrix(df):
    T_k = df["온도1"].values
    u_k = df["u"].values
    I_k = df["I"].values.astype(int)

    u_I1 = u_k * (I_k == 1)
    u_I2 = u_k * (I_k == 2)
    u_I3 = u_k * (I_k == 3)

    X = np.column_stack([T_k, u_I1, u_I2, u_I3])
    y = df["T_next"].values
    return X, y

def fit_ARX1_ab(csv_path):
    df = prepare_dataset(csv_path)
    X, y = build_design_matrix(df)

    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b1, b2, b3 = theta

    y_hat = X @ np.array([a, b1, b2, b3])
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))

    print("==== Fit Result ====")
    print("csv path : %s" %csv_path)
    print(f"a  = {a:.6f}")
    print(f"b = {b1:.6f}")
    if False :
        # 열이 유지되는 구간의 피팅값을 사용하지 X
        print(f"b1 = {b1:.6f}")  
        print(f"b2 = {b2:.6f}")
        print(f"b3 = {b3:.6f}")
    print(f"RMSE = {rmse:.6f}")

    return a, b1, b2, b3, rmse

def fit_ARX1_ab(df):
    df = prepare_dataset(df)
    X, y = build_design_matrix(df)

    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b1, b2, b3 = theta

    y_hat = X @ np.array([a, b1, b2, b3])
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))

    print("==== Fit Result ====")
    print(f"a  = {a:.6f}")
    print(f"b = {b1:.6f}")
    if False :
        # 열이 유지되는 구간의 피팅값을 사용하지 X
        print(f"b1 = {b1:.6f}")  
        print(f"b2 = {b2:.6f}")
        print(f"b3 = {b3:.6f}")
    print(f"RMSE = {rmse:.6f}")

    return a, b1, b2, b3, rmse





if __name__ == "__main__":
    RAW_DATA_PATH_LIST = [
        '/home/junhuiwoo/Desktop/HeatMPC/dat_old/250905-A081.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/dat_old/250904-A083.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/dat_old/250904-A082.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/dat_old/250904-A081.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/dat_old/250907-A082.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/dat_old/250907-A081.csv',
    ]
        
    PRE_DATA_PATH_LIST = [
        '/home/junhuiwoo/Desktop/HeatMPC/dat_old/dat_preprocessed/250905-A081.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/dat_old/dat_preprocessed/250904-A083.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/dat_old/dat_preprocessed/250904-A082.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/dat_old/dat_preprocessed/250904-A081.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/dat_old/dat_preprocessed/250907-A082.csv',
        '/home/junhuiwoo/Desktop/HeatMPC/dat_old/dat_preprocessed/250907-A081.csv',
    ]
    # DATA_PATH = "/home/junhuiwoo/Desktop/HeatMPC/dat/dat_preprocessed/250904-A081.csv"
    # DATA_PATH = "/Users/junhuiwoo/Desktop/HeatMPC/dat/dat_preprocessed/250904-A082.csv"
    # DATA_PATH = "/Users/junhuiwoo/Desktop/HeatMPC/dat/dat_preprocessed/250904-A083.csv"
    # DATA_PATH = "/Users/junhuiwoo/Desktop/HeatMPC/dat/dat_preprocessed/250905-A081.csv"
    # DATA_PATH = "/Users/junhuiwoo/Desktop/HeatMPC/dat/dat_preprocessed/250907-A081.csv"
    # DATA_PATH = "/Users/junhuiwoo/Desktop/HeatMPC/dat/dat_preprocessed/250907-A082.csv"

    for DATA_PATH in RAW_DATA_PATH_LIST : 
        df_prep = load_and_preprocess(DATA_PATH)
        # a, b1, b2, b3, rmse = fit_global_ab(df_prep)
        a, b, *_ = fit_ARX1_ab(df_prep)


