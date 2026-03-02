import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str = "data/Bengaluru_House_Data.csv"):
    """Load and preprocess the Bengaluru house price dataset.

    The notebook exploration revealed the following columns:
    area_type, availability, location, size, society, total_sqft,
    bath, balcony, price.

    Preprocessing steps mirror the EDA and feature engineering:
      * convert total_sqft to numeric (handle ranges/units)
      * extract bedroom_count from the size field
      * impute missing values (median for numerical, mode for categorical)
      * engineer additional features
      * drop high‑missing columns and encode categoricals
      * scale numerical features and split into train/test sets
    """

    df = pd.read_csv(path)
    df = _prepare_df(df)

    X = df.drop('price', axis=1)
    y = df['price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler


# helper functions for reuse ------------------------------------------------

def _convert_total_sqft(value):
    if pd.isna(value):
        return pd.NA
    value = str(value).strip()
    if '-' in value:
        value = value.split('-')[0].strip()
    value = value.replace('Sq. Meter', '').replace('sqft', '').strip()
    try:
        return float(value)
    except ValueError:
        return pd.NA


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard processing steps to a dataframe."""
    # numeric conversions
    df['total_sqft'] = df['total_sqft'].apply(_convert_total_sqft)
    df['bedroom_count'] = df['size'].apply(
        lambda x: int(str(x).split()[0]) if pd.notna(x) else pd.NA
    )

    # engineered features
    df['price_per_sqft'] = df['price'] / (df['total_sqft'] + 1)
    df['price_per_bedroom'] = df['price'] / (df['bedroom_count'] + 1)
    df['bath_bed_ratio'] = df['bath'] / (df['bedroom_count'] + 1)
    df['total_rooms'] = df['bedroom_count'] + df['bath']

    # missing value handling
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype('string')
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    if 'society' in df.columns:
        df = df.drop(columns=['society'])
        if 'society' in cat_cols:
            cat_cols.remove('society')

    # limit cardinality
    for col in list(cat_cols):
        unique_vals = df[col].nunique()
        if unique_vals > 20:
            top = df[col].value_counts().nlargest(10).index
            df[col] = df[col].where(df[col].isin(top), other='Other')

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def preprocess_record(record: dict, feature_columns: list) -> pd.DataFrame:
    """Prepare a single input record into model-ready features.

    ``feature_columns`` should match the columns produced by :func:`load_data`.
    """
    df = pd.DataFrame([record])
    df = _prepare_df(df)
    # drop price if included
    if 'price' in df.columns:
        df = df.drop(columns=['price'])
    # align with training columns
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df
