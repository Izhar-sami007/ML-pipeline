import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(path: str = None) -> pd.DataFrame:
    """Load CSV file or fall back to sklearn iris."""
    if path:
        df = pd.read_csv(path)
    else:
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.frame
        df['species'] = df['target'].map({i: n for i, n in enumerate(data.target_names)})
    return df


def prepare(df: pd.DataFrame, target_col: str = 'species', test_size: float = 0.2, val_size: float = 0.25, random_state: int = 42):
    """Return X_train, X_val, X_test, y_train, y_val, y_test

    val_size is fraction of the remaining after test split (so val ~ 0.25 -> 20% of total when test_size=0.2)
    """
    df = df.copy()
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    # split temp into val and test (val_size fraction of temp)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test