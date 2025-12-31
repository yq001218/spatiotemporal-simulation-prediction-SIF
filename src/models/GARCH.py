from arch import arch_model

class GARCHModel:
    """
    GARCH model for residual volatility
    """

    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q

    def fit(self, residuals):
        self.model = arch_model(
            residuals,
            vol='Garch',
            p=self.p,
            q=self.q
        )
        self.results = self.model.fit(disp="off")

    def predict(self, horizon=1):
        forecast = self.results.forecast(horizon=horizon)
        return forecast.mean.iloc[-1].values
