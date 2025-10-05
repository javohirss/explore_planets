from sklearn.impute import SimpleImputer

def clean_columns_for_ml_alternative(df, target_col='tfopwg_disp'):
    """
    Remove non-predictive columns for ML without creating new features.
    
    Removes:
    - Error/uncertainty columns (err1, err2, errlim)
    - Limit flag columns (lim)
    - String representation columns (str)
    - Metadata/identifier columns
    - URL and reference columns
    
    Returns:
    --------
    df_clean, cols_to_drop
    """
    all_cols = df.columns.tolist()
    cols_to_drop = []
    
    for col in all_cols:
        if col == target_col:
            continue
            
        # Drop error, limit, and string columns
        if (col.endswith('err1') or col.endswith('err2') or 
            col.endswith('errlim') or col.endswith('lim') or 
            col.endswith('str') or col.endswith('url')):
            cols_to_drop.append(col)
        
        # Drop identifier and metadata columns
        if any(x in col.lower() for x in ['rowid', 'htm', 'flag', 'comment', 'ref', 'url']):
            cols_to_drop.append(col)
    
    cols_to_drop = list(set(cols_to_drop))
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"✓ Dropped {len(cols_to_drop)} non-predictive columns")
    print(f"✓ Remaining columns: {len(df_clean.columns)}")
    print(f"  Example dropped: {', '.join(cols_to_drop[:5])}")
    
    return df_clean, cols_to_drop



def prepare_data(df, target_col='tfopwg_disp'):

    df_clean, cols = clean_columns_for_ml_alternative(df, target_col)
    print(df_clean)
    

    X = df_clean.select_dtypes(include=['float64', 'int64'])
    

    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    
    missing_pct = X.isna().sum() / len(X) * 100
    high_missing_cols = missing_pct[missing_pct > 80].index.tolist()
    if high_missing_cols:
        X = X.drop(columns=high_missing_cols)
    

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    return X_imputed, X.columns.tolist()