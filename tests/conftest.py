"""
Fixtures communes pour les tests.
"""
import pytest
import pandas as pd
import numpy as np
import yfinance as yf
from unittest.mock import patch


@pytest.fixture
def sample_prices():
    """Fixture qui fournit un DataFrame de prix pour les tests."""
    # Créer un DataFrame de prix synthétique
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = {
        'AAPL': [100 + i + np.sin(i / 10) * 5 for i in range(100)],
        'MSFT': [200 + i + np.cos(i / 10) * 8 for i in range(100)],
        'GOOGL': [150 + i + np.sin(i / 8) * 10 for i in range(100)]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_returns(sample_prices):
    """Fixture qui fournit les rendements calculés à partir des prix."""
    return sample_prices.pct_change().dropna()


@pytest.fixture
def mock_yf_download():
    """Fixture qui mocke la fonction yfinance.download."""
    # Créer des données de prix synthétiques
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = {
        'Open': [100 + i for i in range(100)],
        'High': [102 + i for i in range(100)],
        'Low': [98 + i for i in range(100)],
        'Close': [101 + i for i in range(100)],
        'Adj Close': [101 + i for i in range(100)],
        'Volume': [1000000 + i * 10000 for i in range(100)]
    }
    mock_data = pd.DataFrame(data, index=dates)

    # Patcher la fonction download
    with patch.object(yf, 'download', return_value=mock_data) as mock_download:
        yield mock_download