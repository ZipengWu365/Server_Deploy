from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, ElasticNet
import numpy as np

class BaseLearner:
    def fit(self, X, y): raise NotImplementedError
    def predict(self, X): raise NotImplementedError

class OLS(BaseLearner):
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

class RidgeLearner(BaseLearner):
    def __init__(self, alpha=1.0, random_state=42, **kwargs):
        self.model = Ridge(alpha=alpha, random_state=random_state, **kwargs)
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

class LassoLearner(BaseLearner):
    def __init__(self, alpha=1.0, random_state=42, **kwargs):
        self.model = Lasso(alpha=alpha, random_state=random_state, **kwargs)
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

class Huber(BaseLearner):
    def __init__(self, epsilon=1.35, max_iter=100, **kwargs):
        self.model = HuberRegressor(epsilon=epsilon, max_iter=max_iter, **kwargs)
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

class ElasticNetLearner(BaseLearner):
    def __init__(self, alpha=1.0, l1_ratio=0.5, random_state=42, **kwargs):
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, **kwargs)
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

LEARNERS = {
    "OLS": OLS,
    "Ridge": RidgeLearner,
    "Lasso": LassoLearner,
    "Huber": Huber,
    "ElasticNet": ElasticNetLearner
}

def get_learner(name: str, params: dict):
    if name not in LEARNERS:
        raise ValueError(f"Unknown learner: {name}")
    return LEARNERS[name](**params)
