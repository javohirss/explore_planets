import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler  # удобно для приведения к [0,1]



def tess_preprocess(df: pd.DataFrame):
    nans = df.isna().sum()
    nans = nans[nans>0].reset_index()
    nans.columns = ["columns", "n_missing"]
    full_nans = nans[nans.n_missing==len(df)]["columns"].to_list()
    nans = nans[~nans["columns"].isin(full_nans)]

    # Безопасное удаление колонок с полными NaN
    if full_nans:
        df.drop(columns=full_nans, axis=1, inplace=True)
    
    dup_cols = [col for col in df.columns if col.endswith("1") or col.endswith("2") or col.endswith("lim") or col.endswith("err")]
    nans = nans[~nans["columns"].isin(dup_cols)]
    
    # Безопасное удаление дублирующих колонок
    if dup_cols:
        df.drop(dup_cols, axis=1, inplace=True)

    # df = df.drop(columns=['rastr', 'decstr', 'toi_created', 'rowupdate', "ra", "dec", "toi", "tid", "ctoi_alias", "toipfx"], axis=1)
    # Безопасное удаление конкретных колонок
    columns_to_drop = ["ra", "dec", "toi", "tid", "toipfx"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df = df.drop(columns=existing_columns_to_drop, axis=1)
    df = df.drop(columns=df.select_dtypes(include=["object"]).columns)

    if "tfopwg_disp" in df.columns:
        tess_map = {
            "CP": 0,
            "KP": 0,
            "PC": 0,
            "APC": 1,
            "FP": 1,
            "FA": 1
        }

        df["target"] = df["tfopwg_disp"].map(tess_map)
        df.drop(columns=["tfopwg_disp"], axis=1, inplace=True)
        y = df["target"]
        X = df.drop(columns=["target"], axis=1)

        return X, y
    
    # Если нет колонки tfopwg_disp, возвращаем только X
    return df, None











def get_reduntant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_reduntant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]