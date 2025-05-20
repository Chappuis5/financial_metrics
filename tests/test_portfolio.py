"""
Tests pour le module portfolio.
"""
import pytest
import pandas as pd
import numpy as np
from portfolio import optimizer, allocator


def test_portfolio_return(sample_returns):
    """Test du calcul du rendement de portefeuille."""
    # Créer des poids égaux
    n = len(sample_returns.columns)
    weights = np.array([1 / n] * n)

    # Calculer le rendement du portefeuille
    result = optimizer.portfolio_return(weights, sample_returns)

    # Vérifier
    assert isinstance(result, float)

    # Calculer manuellement
    expected = np.sum(sample_returns.mean() * weights) * 252
    assert abs(result - expected) < 1e-10


def test_portfolio_volatility(sample_returns):
    """Test du calcul de la volatilité du portefeuille."""
    # Créer des poids égaux
    n = len(sample_returns.columns)
    weights = np.array([1 / n] * n)

    # Calculer la volatilité du portefeuille
    result = optimizer.portfolio_volatility(weights, sample_returns)

    # Vérifier
    assert isinstance(result, float)
    assert result > 0


def test_min_volatility_portfolio(sample_returns):
    """Test de l'optimisation du portefeuille à volatilité minimale."""
    # Trouver le portefeuille à volatilité minimale
    result = optimizer.min_volatility_portfolio(sample_returns)

    # Vérifier la structure
    assert isinstance(result, dict)
    assert "weights" in result
    assert "return" in result
    assert "volatility" in result
    assert "sharpe" in result

    # Vérifier les poids
    weights = result["weights"]
    assert isinstance(weights, pd.Series)
    assert len(weights) == len(sample_returns.columns)
    assert abs(weights.sum() - 1.0) < 1e-10  # Somme des poids = 1
    assert (weights >= 0).all()  # Pas de vente à découvert


def test_max_sharpe_portfolio(sample_returns):
    """Test de l'optimisation du portefeuille à ratio de Sharpe maximal."""
    # Trouver le portefeuille à ratio de Sharpe maximal
    result = optimizer.max_sharpe_portfolio(sample_returns, risk_free_rate=0.01)

    # Vérifier la structure
    assert isinstance(result, dict)
    assert "weights" in result
    assert "return" in result
    assert "volatility" in result
    assert "sharpe" in result

    # Vérifier les poids
    weights = result["weights"]
    assert abs(weights.sum() - 1.0) < 1e-10  # Somme des poids = 1


def test_equal_weight_allocation():
    """Test de l'allocation à poids égaux."""
    # Créer une liste de tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

    # Créer l'allocation
    result = allocator.equal_weight_allocation(tickers)

    # Vérifier
    assert isinstance(result, pd.Series)
    assert len(result) == len(tickers)
    assert result.sum() == 1.0
    assert (result == 0.25).all()  # Poids égaux de 0.25


def test_market_cap_weight_allocation():
    """Test de l'allocation pondérée par capitalisation."""
    # Créer une liste de tickers et capitalisations
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    market_caps = [2000, 1500, 1000]  # en milliards de dollars

    # Créer l'allocation
    result = allocator.market_cap_weight_allocation(tickers, market_caps)

    # Vérifier
    assert isinstance(result, pd.Series)
    assert len(result) == len(tickers)
    assert abs(result.sum() - 1.0) < 1e-10

    # Vérifier manuellement
    total_cap = sum(market_caps)
    expected = pd.Series([cap / total_cap for cap in market_caps], index=tickers)
    pd.testing.assert_series_equal(result, expected)


def test_rebalance_portfolio():
    """Test du rééquilibrage de portefeuille."""
    # Créer des poids actuels et cibles
    current = pd.Series({'AAPL': 0.4, 'MSFT': 0.35, 'GOOGL': 0.25})
    target = pd.Series({'AAPL': 0.33, 'MSFT': 0.33, 'GOOGL': 0.34})

    # Calculer le rééquilibrage
    result = allocator.rebalance_portfolio(current, target, threshold=0.05)

    # Vérifier
    assert isinstance(result, dict)
    assert "need_rebalance" in result
    assert "trades" in result
    assert result["need_rebalance"] == True  # Différences > 5%

    # Vérifier les transactions
    trades = result["trades"]
    assert isinstance(trades, pd.Series)
    assert abs(trades.sum()) < 1e-10  # Somme des transactions = 0

