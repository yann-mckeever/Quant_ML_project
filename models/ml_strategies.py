from models.base_model import FinancialModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from utils.decorators import time_it, log_execution
import pandas as pd
import numpy as np

class RandomForestStrategy(FinancialModel):
    """  
    Classe enfant implémentée une stratégie basée sur une fôret aléatoire

    """

    def __init__(self, n_estimators: int=100, max_depth: int=100):
        #On appelle le constructeur de la classe mère pour lui donner un nom:
        super().__init__(model_name="Random Forest Classifier")
        #On instancie le vrai modèle avec scikit-learn
        self.model=RandomForestClassifier(n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          random_state=42)
    @log_execution
    @time_it
    def train(self, X_train:pd.DataFrame,y_train:pd.Series)->None:
        """  
        Implémentation du modèle d'enraînement par la classe mère
        """
        self.model.fit(X_train,y_train)
        self.is_trained=True

    def predict(self, X:pd.DataFrame)->np.ndarray:
        """  
        Implémentation de la prédiction
        """
        if not self.is_trained :
            raise ValueError(f"Impossible de prédire le modèle : {self.model_name} n'est pas entraîné")
        return self.model.predict(X)
        
class LogisticRegressionStrategy(FinancialModel):
    """  
    Classe enfant implémentée d'une stratégie basée sur une régréssion logistique
    """
    def __init__(self, penalty:str='l2', C:float=1.0):
        """  
        Penalty l1 pour lasso
                l2 pour Ridge
        C: inverse de la fonction de régularisation plus C est petit, plus la contrainte est forte
        """
        super().__init__(model_name=f"Logistic Regression ({penalty.upper()})")
        #Note : le solveur liblinear est nécéssaire pour gérer les pénalités Lasso
        self.model= LogisticRegression(penalty=penalty,C=C,solver='liblinear',random_state=42)

    @log_execution
    @time_it
    def train(self, X_train:pd.DataFrame, y_train:pd.Series)->None:
        self.model.fit(X_train,y_train)
        self.is_trained=True

    def predict(self, X:pd.DataFrame)->np.ndarray:
        if not self.is_trained:
            raise ValueError(f"Impossible de prédire : le modèle f{self.model_name} n'est pas entraîné")
        return self.model.predict(X)
        