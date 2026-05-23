from core.data_manager import DataManager
from models.ml_strategies import RandomForestStrategy, LogisticRegressionStrategy
from backtest.engine import BacktestEngine
from sklearn.preprocessing import StandardScaler

def main():
    print("[SYSTEM] Initialisation de l'application QuantML")

    #Pipeline de données (Téléchargement de l'ETF Cac40)
    dm=DataManager(ticker="^FCHI", start_date="2020-01-01", end_date="2026-05-05")
    dm.download_data()
    dm.add_features()
    X,y=dm.get_X_y()
    
    #Séparation Train/Test temporelle 80% train 20% test

    split_index=int(len(X)*0.8)
    X_train, X_test= X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test= y.iloc[:split_index], y.iloc[split_index:]

    market_test_returns=X_test['return']

    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    #strategy=RandomForestStrategy(n_estimators=150, max_depth=5)
    #strategy=RandomForestStrategy(n_estimators=150, max_depth=3)
    #strategy=LogisticRegressionStrategy(penalty='l1', C=0.1)
    strategy=LogisticRegressionStrategy(penalty='l2', C=0.5)
    strategy.train(X_train_scaled,y_train)

    predictions=strategy.predict(X_test_scaled)

    engine=BacktestEngine(initial_capital=10000.0)
    engine.run(market_test_returns,predictions)

    engine.compute_metrics()
    engine.plot_performance(ticker_name='^FCHI')

if __name__=="__main__":
    main()
