from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, Y_train):
        pass

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def coef(self):
        pass



class Logistic_Regression_Baseline(BaseModel):
    def __init__(self, C=1.0, max_iter=100):
        self.model = LogisticRegression(
            C=C, max_iter=max_iter
        )
        self._is_fitted = False

    def fit(self, X_train, Y_train):
        self._is_fitted = True
        return self.model.fit(X_train, Y_train)
    
    def predict(self,X):
        if not self._is_fitted:
            raise ValueError("Vectorizer is not fitted yet. Call fit() first.")
        return self.model.predict(X)
    
    def coef(self):
        if not self._is_fitted:
            raise ValueError("Vectorizer is not fitted yet. Call fit() first.")
        return self.model.coef_
        


    