
from sklearn.ensemble import RandomForestRegressor

class RFModel:
    """
    Random Forest baseline
    """

    def __init__(self,
                 n_estimators=300,
                 max_depth=None,
                 random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X, y):
        X_flat = X.reshape(len(X), -1)
        self.model.fit(X_flat, y)

    def predict(self, X):
        X_flat = X.reshape(len(X), -1)
        return self.model.predict(X_flat)
