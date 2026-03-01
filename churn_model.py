import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_model():

    df = pd.read_csv("churn-bigml-20.csv")
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Separate categorical and numeric columns
    categorical_cols = X.select_dtypes(include="object").columns
    numeric_cols = X.select_dtypes(exclude="object").columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # Pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, X.columns, accuracy, X_test, y_test