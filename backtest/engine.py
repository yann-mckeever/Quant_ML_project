import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.decorators import time_it, log_execution

class BacktestEngine:
    def __init__(self,initial_capital :float =10000.0):
        """  
        Initialise le moteur du backtest avec un capital de départ
        """
        self.initial_capital=initial_capital
        self.results=None
    
    @log_execution
    @time_it
    def run(self, market_returns : pd.Series, model_prediction:np.ndarray) ->pd.DataFrame:
        """  
        Simule la stratégie de trading basée sur les prédictions du modèle

        market_returns : Les rendements réels logarithmiques du marché sur la période de test
        model_prediction : Le tableau de 0 et 1 généré par le modèle
        """

        if len(market_returns)!=len(model_prediction):
            raise ValueError("Le nombre de rendements de marché et de prédictions doit être identique")
        
        #On crée un DataFrame pour aligner proprement les séries chronologiques
        self.results=pd.DataFrame(index=market_returns.index)
        self.results['market_return']=market_returns
        self.results['signal']=np.where(model_prediction==1,1,-1)

        #Rendement de notre stratégie : 
        # Si signal =1, on gagne le rendement du marché, Si signal =0 on fait 0%

        self.results['strategy_return']=self.results['signal'].shift(1)*self.results['market_return']
        self.results.dropna(inplace=True)

        #On calcule la performance cumulée : passage du log au réel

        self.results['cum_market_return']=np.exp(self.results['market_return'].cumsum())
        self.results['cum_strategy_return']=np.exp(self.results['strategy_return'].cumsum())

        #Evolution de notre capital

        self.results['portfolio_value']=self.initial_capital*self.results['cum_strategy_return']

        return self.results
    
    def compute_metrics(self):
        """  
        Calcule les indicateurs de performance clés (KPIs) pour l'entretien
        """

        if self.results is None:
            raise ValueError("Le backtest doit être éxécuté avant de calculer les metrics")
        
        #Rendement total du marché vs. rendement de la stratégie :

        total_market_perf=(self.results['cum_market_return'].iloc[-1]-1)*100
        total_strategy_perf=(self.results['cum_strategy_return'].iloc[-1]-1)*100
        final_capital=self.results['portfolio_value'].iloc[-1]

        print("\n" + "="*35)
        print("     RESULTATS DU BACKTEST       ")
        print("="*35)

        print(f"Capital de départ       : {self.initial_capital:,.2f} $")
        print(f"Capital final           : {final_capital:,.2f} $")
        print(f"Performance strategie   : {total_strategy_perf:+.2f} %")
        print(f"Performance du marché (B&H)  : {total_market_perf:+.2f} %")

        return total_strategy_perf, total_market_perf
    
    def plot_performance(self, ticker_name:str):
        """  
        Génère un graphique professionel comparant les deux courbes de rendement
        """

        if self.results is None :
            raise ValueError("Aucun graphique à afficher lancez le backtest d'abord")
        
        plt.figure(figsize=(12,6))
        plt.plot(self.results['cum_market_return'], label=f"Marché ({ticker_name}) - Buy & Hold", color='gray', alpha=0.7)
        plt.plot(self.results['cum_strategy_return'], label="Stratégie machine learning", color='darkgreen', linewidth=2)
        plt.title(f"Simulation de Performance Historique - {ticker_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Multiplicateur de gain (Base 1.0)")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.show()




