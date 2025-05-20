"""
Tests pour le module API.
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Importer l'application FastAPI
from api.endpoints import app

# Créer un client de test
client = TestClient(app)

def test_read_root():
    """Test de l'endpoint racine."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data

# Correction pour le test fetch_data_endpoint
@patch('data.fetchers.fetch_data')
def test_fetch_data_endpoint(mock_fetch_data):
    """Test de l'endpoint /fetch_data/."""
    # Configurer le mock avec des données sérialisables
    mock_data = pd.Series([100, 101, 102], index=pd.date_range('2020-01-01', periods=3))
    mock_fetch_data.return_value = mock_data

    # Faire la requête
    response = client.post("/fetch_data/", json={"ticker": "AAPL", "period": "1mo"})

    # Vérifier
    assert response.status_code == 200
    data = response.json()
    assert "ticker" in data
    assert data["ticker"] == "AAPL"
    assert "prices" in data

# Correction pour le test analyze_asset_endpoint
@patch('data.fetchers.fetch_data')
@patch('data.processors.calculate_returns')
def test_analyze_asset_endpoint(mock_calculate_returns, mock_fetch_data):
    """Test de l'endpoint /analyze_asset/."""
    # Configurer les mocks correctement
    mock_prices = pd.Series([100, 102, 101, 103, 105], index=pd.date_range('2020-01-01', periods=5))
    mock_returns = pd.Series([0.02, -0.01, 0.02, 0.019], index=pd.date_range('2020-01-02', periods=4))

    # S'assurer que les mocks retournent les bonnes valeurs
    mock_fetch_data.return_value = mock_prices
    mock_calculate_returns.return_value = mock_returns

    # Faire la requête
    response = client.post("/analyze_asset/", json={"ticker": "AAPL", "period": "1mo"})

    # Vérifier la réponse
    assert response.status_code == 200
    data = response.json()
    assert "ticker" in data
    assert data["ticker"] == "AAPL"
    assert "metrics" in data
    assert "geometric_return" in data["metrics"]

# Correction pour le test optimize_portfolio_endpoint
@patch('data.fetchers.fetch_data')
@patch('data.processors.calculate_returns')
@patch('portfolio.optimizer.min_volatility_portfolio')
@patch('portfolio.optimizer.max_sharpe_portfolio')
def test_optimize_portfolio_endpoint(mock_max_sharpe, mock_min_vol, mock_calculate_returns, mock_fetch_data):
    """Test de l'endpoint /optimize_portfolio/."""
    # Configurer les mocks correctement
    dates = pd.date_range('2020-01-01', periods=100)
    mock_prices = pd.DataFrame({
        'AAPL': [100 + i + np.sin(i/10) * 5 for i in range(100)],
        'MSFT': [200 + i + np.cos(i/10) * 8 for i in range(100)]
    }, index=dates)

    mock_returns = mock_prices.pct_change().dropna()

    # Configurer les résultats d'optimisation
    mock_min_vol.return_value = {
        "weights": pd.Series([0.6, 0.4], index=['AAPL', 'MSFT']),
        "return": 0.12,
        "volatility": 0.15,
        "sharpe": 0.8
    }

    mock_max_sharpe.return_value = {
        "weights": pd.Series([0.5, 0.5], index=['AAPL', 'MSFT']),
        "return": 0.14,
        "volatility": 0.18,
        "sharpe": 0.9
    }

    # Configurer les mocks de données
    mock_fetch_data.return_value = mock_prices
    mock_calculate_returns.return_value = mock_returns

    # Faire la requête
    response = client.post("/optimize_portfolio/",
                          json={"tickers": ["AAPL", "MSFT"], "period": "1y", "risk_free_rate": 0.01})

    # Vérifier la réponse
    assert response.status_code == 200
    data = response.json()
    assert "tickers" in data
    assert "min_volatility_portfolio" in data
    assert "max_sharpe_portfolio" in data
    assert "weights" in data["min_volatility_portfolio"]
    assert "weights" in data["max_sharpe_portfolio"]