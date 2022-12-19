import pandas as pd

def filter_features(df: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
    """ Filters the dataframe by the threshold of missing values

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be filtered
    threshold : float
        The threshold to filter the dataframe

    Returns
    -------
    pd.DataFrame
        The filtered dataframe
    """
    df_copy = df.copy()
    df_copy = df_copy.isna().sum()/len(df_copy)
    df_copy = df_copy > threshold
    features_to_drop = df_copy[df_copy].index.to_list()
    return df.drop(columns=features_to_drop)

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """ Removes the outliers from the dataframe using the IQR method

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be filtered

    Returns
    -------
    pd.DataFrame
        The filtered dataframe
    """
    df_copy = df.copy()
    q1 = df_copy.quantile(0.25)
    q3 = df_copy.quantile(0.75)
    iqr = q3 - q1
    df_copy = df_copy[~((df_copy < (q1 - 1.5 * iqr)) | (df_copy > (q3 + 1.5 * iqr))).any(axis=1)]
    return df_copy

