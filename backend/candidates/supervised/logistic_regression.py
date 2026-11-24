from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_candidate():

    model = Pipeline([
        ("scaler", StandardScaler()),

        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ))
    ])

    return model