from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class FinancialModel(ABC):
    """ 
    Classe mère abstraite
    Interface pour que tous nos modèles ML marchent sur la même structure
    Similaire à virtual void en C++
    """
    def __init__(self, model_name:str):
        self.model_name=model_name
        self.is_trained=False

    @abstractmethod
    def train(self, X_train:pd.DataFrame,y_train=pd.Series)->None :
        """  
        Méthode obligatoire pour entraîner un modèle
        Chaque classe enfin devra l'implémenter avec son propre algorithme
        """
        pass

    @abstractmethod
    def predict(self, X:pd.DataFrame)->np.ndarray:
        """
        Méthode obligatoire pour générer les prédictions 0 ou 1
        """
        pass
    
    def __str__(self):
        """  
        Méthode spéciale (dunder) pour afficher le nom d'un modèle.

        """
        return f"Modèle financier: {self.model_name} [Entrainé :{self.is_trained}]"
    


