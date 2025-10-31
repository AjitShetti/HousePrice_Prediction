import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str = "data/Housing.csv"):
    df = pd.read_csv(path)

    binary_cols = ['mainroad', 'guestroom', 'basement',
                   'hotwaterheating', 'airconditioning', 'prefarea']
    
    for col in binary_cols:
        df[col] = df[col].str.lower().map({'yes':1, 'no':0})

    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    X = df.drop('price', axis=1)
    y = df['price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler
