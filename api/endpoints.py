"""
Points d'entrée API utilisant FastAPI.
"""
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import json

from data import fetchers, processors
from metrics import returns, risk, ratios
from portfolio import optimizer, allocator

# Initialisation de l'application FastAPI
app = FastAPI(title="Financial Metrics API", 
             description="API pour l'analyse et l'optimisation de portefeuilles d'investissement")

# Définition des modèles de données
class TickerData(BaseModel):
    ticker: str
    period: str = "5y"
    interval: str = "1d"

class PortfolioRequest(BaseModel):
    tickers: List[str]
    period: str = "5y"
    interval: str = "1d"
    risk_free_rate: float = 0.0

class AllocationRequest(BaseModel):
    tickers: List[str]
    allocation_type: str = "equal"  # equal, market_cap, risk_parity
    weights: Optional[List[float]] = None

# Endpoints
@app.get("/")
def read_root():
    return {"message": "Financial Metrics API"}

@app.post("/fetch_data/")
def api_fetch_data(data: TickerData):
    """Récupère les données historiques pour un ticker."""
    prices = fetchers.fetch_data(data.ticker, period=data.period, interval=data.interval)
    
    # Convertir à un format serializable
    result = {
        "ticker": data.ticker,
        "prices": prices.dropna().to_dict()
    }
        
    return result

@app.post("/allocate_portfolio/")
def api_allocate_portfolio(data: AllocationRequest):
    """Alloue les actifs selon la stratégie choisie."""
    # Vérifier les poids fournis
    if data.weights and len(data.weights) != len(data.tickers):
        return {"error": "Le nombre de poids doit correspondre au nombre de tickers"}
    
    # Stratégies d'allocation
    if data.allocation_type == "equal":
        weights = allocator.equal_weight_allocation(data.tickers)
    elif data.allocation_type == "market_cap":
        # Obtenir les capitalisations boursières
        market_caps = []
        for ticker in data.tickers:
            info = fetchers.get_asset_info(ticker)
            market_cap = info.get('marketCap', 0)
            market_caps.append(market_cap if market_cap != 'N/A' else 0)
        
        weights = allocator.market_cap_weight_allocation(data.tickers, market_caps)
    elif data.allocation_type == "risk_parity":
        # Pour la parité de risque, nous avons besoin des données historiques
        prices = fetchers.fetch_data(data.tickers, period=data.period, interval=data.interval)
        rets = processors.calculate_returns(prices)
        weights = allocator.risk_parity_allocation(rets)
    elif data.allocation_type == "custom" and data.weights:
        # Allocation personnalisée
        weights = pd.Series(data.weights, index=data.tickers)
        weights = weights / weights.sum()  # Normaliser
    else:
        return {"error": "Type d'allocation non reconnu"}
    
    # Analyser le portefeuille avec ces poids
    prices = fetchers.fetch_data(data.tickers, period=data.period, interval=data.interval)
    rets = processors.calculate_returns(prices)
    
    # Calculer les métriques du portefeuille
    portfolio_ret = (rets * weights).sum(axis=1)
    
    metrics = {
        "return": returns.geometric_mean_return(portfolio_ret),
        "volatility": risk.volatility(portfolio_ret),
        "sharpe_ratio": ratios.sharpe_ratio(portfolio_ret),
        "sortino_ratio": ratios.sortino_ratio(portfolio_ret)
    }
    
    return {
        "allocation_type": data.allocation_type,
        "tickers": data.tickers,
        "weights": weights.to_dict(),
        "metrics": metrics
    }

@app.post("/calculate_metrics/")
def api_calculate_metrics(data: PortfolioRequest):
    """Calcule toutes les métriques pour un ensemble d'actifs."""
    # Récupérer les données
    prices = fetchers.fetch_data(data.tickers, period=data.period, interval=data.interval)
    
    # Calculer les rendements
    rets = processors.calculate_returns(prices)
    
    # Initialiser les résultats
    results = {"tickers": {}}
    
    # Calculer les métriques pour chaque ticker
    for ticker in data.tickers:
        if ticker in rets.columns:
            ticker_rets = rets[ticker]
            ticker_prices = prices[ticker]
            
            results["tickers"][ticker] = {
                "geometric_return": returns.geometric_mean_return(ticker_rets),
                "arithmetic_return": returns.arithmetic_mean_return(ticker_rets),
                "volatility": risk.volatility(ticker_rets),
                "max_drawdown": risk.max_drawdown(ticker_prices),
                "downside_risk": risk.downside_risk(ticker_rets),
                "var_95": risk.value_at_risk(ticker_rets, 0.05),
                "sortino_ratio": ratios.sortino_ratio(ticker_rets),
                "sharpe_ratio": ratios.sharpe_ratio(ticker_rets),
                "calmar_ratio": ratios.calmar_ratio(ticker_rets, ticker_prices)
            }
    
    return results

# Fonction de démarrage de l'API
def start_api(host="0.0.0.0", port=8000):
    """Démarre l'API FastAPI."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_api()

@app.post("/analyze_asset/")
def api_analyze_asset(data: TickerData):
    """Analyse un actif et calcule les métriques principales."""
    # Récupérer les données
    prices = fetchers.fetch_data(data.ticker, period=data.period, interval=data.interval)
    
    if len(prices) == 0:
        return {"error": "Impossible de récupérer les données pour ce ticker"}
    
    # Calculer les rendements
    rets = processors.calculate_returns(prices)
    
    # Calculer les métriques
    analysis = {
        "ticker": data.ticker,
        "period": data.period,
        "metrics": {
            "geometric_return": returns.geometric_mean_return(rets),
            "volatility": risk.volatility(rets),
            "max_drawdown": risk.max_drawdown(prices),
            "sharpe_ratio": ratios.sharpe_ratio(rets),
            "sortino_ratio": ratios.sortino_ratio(rets),
            "calmar_ratio": ratios.calmar_ratio(rets, prices)
        }
    }
    
    return analysis

@app.post("/optimize_portfolio/")
def api_optimize_portfolio(data: PortfolioRequest):
    """Trouve le portefeuille optimal selon différents critères."""
    # Récupérer les données pour tous les tickers
    all_prices = fetchers.fetch_data(data.tickers, period=data.period, interval=data.interval)
    
    # Calculer les rendements
    all_returns = processors.calculate_returns(all_prices)
    
    # Optimiser le portefeuille
    min_vol_portfolio = optimizer.min_volatility_portfolio(all_returns)
    max_sharpe_portfolio = optimizer.max_sharpe_portfolio(all_returns, data.risk_free_rate)
    
    # Préparer le résultat
    result = {
        "tickers": data.tickers,
        "min_volatility_portfolio": {
            "weights": min_vol_portfolio["weights"].to_dict(),
            "return": min_vol_portfolio["return"],
            "volatility": min_vol_portfolio["volatility"],
            "sharpe": min_vol_portfolio["sharpe"]
        },
        "max_sharpe_portfolio": {
            "weights": max_sharpe_portfolio["weights"].to_dict(),
            "return": max_sharpe_portfolio["return"],
            "volatility": max_sharpe_portfolio["volatility"],
            "sharpe": max_sharpe_portfolio["sharpe"]
        }
    }

    return result
