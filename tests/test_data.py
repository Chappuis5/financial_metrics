"""
Tests pour le module data.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from data import fetchers, processors


def test_fetch_data(mock_yf_download):
    """Test de la fonction fetch_data."""
    # Tester la récupération des données pour un seul ticker
    result = fetchers.fetch_data('AAPL', period='5d')

    # Vérifier que yf.download a été appelé avec les bons paramètres
    mock_yf_download.assert_called_once()
    call_args = mock_yf_download.call_args[1]
    assert call_args['period'] == '5d'

    # Vérifier le résultat
    assert isinstance(result, pd.Series)


def test_fetch_etf_data(mock_yf_download):
    """Test de la fonction fetch_etf_data."""
    # Tester la récupération des données ETF
    result = fetchers.fetch_etf_data(['SPY', 'QQQ'], period='1y')

    # Vérifier l'appel
    mock_yf_download.assert_called_once()

    # Vérifier le résultat
    assert mock_yf_download.called


def test_clean_data(sample_prices):
    """Test de la fonction clean_data."""
    # Ajouter quelques valeurs NaN
    df_with_nans = sample_prices.copy()
    df_with_nans.iloc[5:10, 0] = np.nan

    # Nettoyer les données
    clean_df = processors.clean_data(df_with_nans)

    # Vérifier que les NaN ont été supprimés
    assert not clean_df.isnull().any().any()
    assert len(clean_df) < len(df_with_nans)


def test_calculate_returns(sample_prices):
    """Test de la fonction calculate_returns."""
    # Calculer les rendements
    rets = processors.calculate_returns(sample_prices)

    # Vérifier que les rendements sont corrects
    expected_rets = sample_prices.pct_change().dropna()
    pd.testing.assert_frame_equal(rets, expected_rets)


def test_resample_data(sample_prices):
    """Test de la fonction resample_data."""
    # Rééchantillonner les données quotidiennes en mensuelles
    monthly = processors.resample_data(sample_prices, freq='M')

    # Vérifier que le nombre de périodes est correct
    assert len(monthly) < len(sample_prices)
    assert monthly.index.freq == 'M'