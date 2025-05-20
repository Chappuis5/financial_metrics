"""
Module pour le calcul des ratios financiers.
"""
import numpy as np
import pandas as pd
from financial_metrics.metrics.returns import arithmetic_mean_return
from financial_metrics.metrics.risk import volatility, downside_risk, max_drawdown

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calcule le ratio de Sharpe (rendement ajusté au risque).
    
    Args:
        returns (Series or DataFrame): Rendements quotidiens
        risk_free_rate (float): Taux sans risque annualisé
        periods_per_year (int): Nombre de périodes par an
    
    Returns:
        float or Series: Ratio de Sharpe
    """
    # Convertir le taux sans risque à la même fréquence que les rendements
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculer le ratio de Sharpe
    excess_return = arithmetic_mean_return(returns - rf_daily, annualize=True, periods_per_year=periods_per_year)
    volatility_annual = volatility(returns, annualize=True, periods_per_year=periods_per_year)
    
    return excess_return / volatility_annual

def sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calcule le ratio de Sortino (rendement ajusté au risque à la baisse).
    
    Args:
        returns (Series or DataFrame): Rendements quotidiens
        risk_free_rate (float): Taux sans risque annualisé
        periods_per_year (int): Nombre de périodes par an
    
    Returns:
        float or Series: Ratio de Sortino
    """
    # Convertir le taux sans risque à la même fréquence que les rendements
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculer le ratio de Sortino
    excess_return = arithmetic_mean_return(returns - rf_daily, annualize=True, periods_per_year=periods_per_year)
    downside_volatility = downside_risk(returns, annualize=True, periods_per_year=periods_per_year)
    
    return excess_return / downside_volatility

def calmar_ratio(returns, prices, periods_per_year=252):
    """
    Calcule le ratio de Calmar (rendement annualisé / drawdown maximal).
    
    Args:
        returns (Series or DataFrame): Rendements quotidiens
        prices (Series or DataFrame): Série de prix correspondant aux rendements
        periods_per_year (int): Nombre de périodes par an
    
    Returns:
        float or Series: Ratio de Calmar
    """
    annual_return = arithmetic_mean_return(returns, annualize=True, periods_per_year=periods_per_year)
    mdd = abs(max_drawdown(prices))
    
    # Éviter la division par zéro
    if isinstance(mdd, pd.Series):
        mdd[mdd == 0] = np.nan
    elif mdd == 0:
        return np.nan
        
    return annual_return / mdd

def information_ratio(returns, benchmark_returns, periods_per_year=252):
    """
    Calcule le ratio d'information (alpha / tracking error).
    
    Args:
        returns (Series): Rendements quotidiens
        benchmark_returns (Series): Rendements quotidiens du benchmark
        periods_per_year (int): Nombre de périodes par an
    
    Returns:
        float: Ratio d'information
    """
    # Assurer l'alignement des données
    common_index = returns.index.intersection(benchmark_returns.index)
    returns = returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    
    # Calculer le rendement excédentaire
    excess_returns = returns - benchmark_returns
    
    # Tracking error (annualisé)
    tracking_error = volatility(excess_returns, annualize=True, periods_per_year=periods_per_year)
    
    # Rendement excédentaire moyen (annualisé)
    excess_return_mean = arithmetic_mean_return(excess_returns, annualize=True, periods_per_year=periods_per_year)
    
    return excess_return_mean / tracking_error

