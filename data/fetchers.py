"""
Module pour la récupération des données financières via Yahoo Finance API.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def fetch_data(tickers, period="5y", interval="1d"):
    """
    Récupère les données historiques pour une liste de tickers.
    
    Args:
        tickers (str or list): Symboles boursiers séparés par des espaces ou liste
        period (str): Période (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Intervalle (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        DataFrame: Données historiques avec prix ajustés
    """
    try:
        data = yf.download(tickers, period=period, interval=interval, auto_adjust=True, 
                          progress=False, group_by='ticker')
        
        # Réorganisation si plusieurs tickers
        if isinstance(tickers, list) and len(tickers) > 1:
            return data
        else:
            # Si un seul ticker, récupérer seulement les prix de clôture ajustés
            if 'Adj Close' in data.columns:
                return data['Adj Close']
            else:
                return data['Close']
    except Exception as e:
        print(f"Erreur lors de la récupération des données: {e}")
        return pd.DataFrame()

def fetch_etf_data(etf_symbols, period="5y"):
    """
    Récupère spécifiquement les données d'ETF.
    """
    return fetch_data(etf_symbols, period=period)

def get_asset_info(ticker):
    """
    Récupère les informations de base sur un actif.
    """
    try:
        asset = yf.Ticker(ticker)
        info = asset.info
        return {
            'name': info.get('shortName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'currency': info.get('currency', 'USD'),
            'country': info.get('country', 'N/A')
        }
    except Exception as e:
        print(f"Erreur lors de la récupération des informations: {e}")
        return {}

