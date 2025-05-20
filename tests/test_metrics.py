"""
Tests pour le module metrics.
"""
import pytest
import pandas as pd
import numpy as np
from metrics import returns, risk, ratios


def test_arithmetic_mean_return(sample_returns):
    """Test du calcul du rendement arithmétique moyen."""
    # Calculer le rendement arithmétique
    result = returns.arithmetic_mean_return(sample_returns, annualize=True)

    # Vérifier que le résultat est cohérent
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_returns.columns)

    # Calculer manuellement pour comparer
    expected = sample_returns.mean() * 252
    pd.testing.assert_series_equal(result, expected)


def test_geometric_mean_return(sample_returns):
    """Test du calcul du rendement géométrique moyen."""
    # Calculer le rendement géométrique
    result = returns.geometric_mean_return(sample_returns, annualize=False)

    # Vérifier que le résultat est cohérent
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_returns.columns)

    # Vérifier manuellement le résultat pour un actif
    ticker = sample_returns.columns[0]
    manual_calc = (1 + sample_returns[ticker]).prod() ** (1 / len(sample_returns)) - 1
    assert abs(result[ticker] - manual_calc) < 1e-10


def test_cumulative_returns(sample_returns):
    """Test du calcul des rendements cumulatifs."""
    # Calculer les rendements cumulatifs
    result = returns.cumulative_returns(sample_returns)

    # Vérifier les dimensions
    assert result.shape == sample_returns.shape

    # Vérifier manuellement
    ticker = sample_returns.columns[0]
    expected = (1 + sample_returns[ticker]).cumprod() - 1
    pd.testing.assert_series_equal(result[ticker], expected)


def test_volatility(sample_returns):
    """Test du calcul de la volatilité."""
    # Calculer la volatilité
    result = risk.volatility(sample_returns, annualize=True)

    # Vérifier
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_returns.columns)

    # Calculer manuellement
    expected = sample_returns.std() * np.sqrt(252)
    pd.testing.assert_series_equal(result, expected)


def test_max_drawdown(sample_prices):
    """Test du calcul du drawdown maximal."""
    # Calculer le drawdown maximal
    result = risk.max_drawdown(sample_prices)

    # Vérifier
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_prices.columns)
    assert (result <= 0).all()  # Les drawdowns sont négatifs


def test_sharpe_ratio(sample_returns):
    """Test du calcul du ratio de Sharpe."""
    # Calculer le ratio de Sharpe
    result = ratios.sharpe_ratio(sample_returns, risk_free_rate=0.0)

    # Vérifier
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_returns.columns)


def test_sortino_ratio(sample_returns):
    """Test du calcul du ratio de Sortino."""
    # Calculer le ratio de Sortino
    result = ratios.sortino_ratio(sample_returns, risk_free_rate=0.0)

    # Vérifier
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_returns.columns)


def test_calmar_ratio(sample_returns, sample_prices):
    """Test du calcul du ratio de Calmar."""
    # Calculer le ratio de Calmar
    result = ratios.calmar_ratio(sample_returns, sample_prices)

    # Vérifier
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_returns.columns)
