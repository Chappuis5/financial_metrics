"""
Tests d'intégration pour la librairie financial_metrics.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from data import fetchers, processors
from metrics import returns, risk, ratios
from portfolio import optimizer, allocator


def test_basic_analysis_flow(mock_yf_download, sample_prices):
    """Test du flux d'analyse de base."""
    # Configurer le mock pour retourner des données connues
    mock_yf_download.return_value = sample_prices

    # Récupérer les données
    prices = fetchers.fetch_data(['AAPL', 'MSFT', 'GOOGL'], period='1y')

    # Calculer les rendements
    rets = processors.calculate_returns(prices)

    # Calculer les métriques de base
    geo_ret = returns.geometric_mean_return(rets)
    vol = risk.volatility(rets)
    mdd = risk.max_drawdown(prices)
    sharpe = ratios.sharpe_ratio(rets)
    sortino = ratios.sortino_ratio(rets)

    # Vérifier que les métriques sont calculées correctement
    assert isinstance(geo_ret, pd.Series)
    assert isinstance(vol, pd.Series)
    assert isinstance(mdd, pd.Series)
    assert isinstance(sharpe, pd.Series)
    assert isinstance(sortino, pd.Series)


def test_portfolio_optimization_flow(sample_returns):
    """Test du flux d'optimisation de portefeuille."""
    # Optimiser le portefeuille
    min_vol = optimizer.min_volatility_portfolio(sample_returns)
    max_sharpe = optimizer.max_sharpe_portfolio(sample_returns, risk_free_rate=0.02)

    # Créer différentes allocations
    tickers = sample_returns.columns.tolist()
    equal_weights = allocator.equal_weight_allocation(tickers)

    # Simuler une allocation de marché
    market_caps = [2000, 1500, 1000]  # Valeurs fictives
    market_weights = allocator.market_cap_weight_allocation(tickers, market_caps)

    # Vérifier le rééquilibrage
    rebalance = allocator.rebalance_portfolio(equal_weights, market_weights, threshold=0.05)

    # Vérifications
    assert isinstance(min_vol, dict)
    assert isinstance(max_sharpe, dict)
    assert isinstance(equal_weights, pd.Series)
    assert isinstance(market_weights, pd.Series)
    assert isinstance(rebalance, dict)
    assert "need_rebalance" in rebalance

    # Vérifier que les poids sont valides
    assert abs(min_vol["weights"].sum() - 1.0) < 1e-10
    assert abs(max_sharpe["weights"].sum() - 1.0) < 1e-10
    assert abs(equal_weights.sum() - 1.0) < 1e-10
    assert abs(market_weights.sum() - 1.0) < 1e-10