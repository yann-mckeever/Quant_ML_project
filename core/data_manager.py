import yfinance as yf
import pandas as pd
import numpy as np 
from utils.decorators import time_it, require_dataframe, log_execution


class DataManager:
    #Constructeur de la classe -> équivalent à C++
    def __init__(self, ticker: str, start_date: str, end_date: str):
        """ 
        Initialise le gestionnaire de données.
        Exemple de ticker : 'GLD' : Or, 'AAPL' : Apple, '^FCHI': CAC40 
           
        """

        self.ticker=ticker
        self.start_date= start_date
        self.end_date=end_date
        self.data=None

    @log_execution
    @time_it
    @require_dataframe
    def download_data(self) -> pd.DataFrame :
        """  
        Télécharge les données historiques depuis Yahoo finance
        """
        print(f"[DATA] Téléchargement des données pour {self.ticker}")
        df=yf.download(self.ticker, start=self.start_date, end=self.end_date)

        if 'Adj Close' in df.columns:
            self.data=df[['Adj Close']].copy()
        else :
            self.data=df[['Close']].copy()

        self.data.columns=['price']
        return self.data
    
    @time_it
    @require_dataframe
    def add_features(self)->pd.DataFrame :
        """  
        Calcul les indicateurs techniques (features) pour le ML
        """
        if self.data is None or self.data.empty :
            raise ValueError("Aucune donnée disponible. Exécutez download_data() d'abord.")
        
        self.data['return']=np.log(self.data['price']/self.data['price'].shift(1))

        self.data['MA_5']=self.data['price'].rolling(window=5).mean()
        self.data['MA_10']=self.data['price'].rolling(window=10).mean()

        self.data['volatility_10']=self.data['return'].rolling(window=10).std()

        #Création de la CIBLE : target 'y' : le marché monte (1) ou baisse/stagne (0) le jour suivant 
        #On décale les rendements de -1 pour projeter le futur sur la ligne du présent
        self.data['target']=np.where(self.data['return'].shift(-1)>0,1,0)

        # Nettoyage : Les calculs glissants (rolling) et décalés (shift) créent des valeurs manquantes (NaN)
        self.data.dropna(inplace=True)

        return self.data
    
    def get_X_y(self):
        """  
        Sépare le DataFrame en matrice de caractéristiques X et vecteur caractéristique y
        """

        if self.data is None or 'target' not in self.data.columns :
            raise ValueError("Les caractéristiques et la cible n'ont pas encore était générée")
        
        feature_cols=['return','MA_5','MA_10','volatility_10']
        X=self.data[feature_cols]
        y=self.data['target']

        return X,y
    

