"""
Module pour le calcul des rendements.
"""
import numpy as np
import pandas as pd

def arithmetic_mean_return(returns, annualize=True, periods_per_year=252):
    """
    Calcule le rendement moyen arithmétique.
    
    Args:
        returns (Series or DataFrame): Rendements quotidiens
        annualize (bool): Si True, annualise le résultat
        periods_per_year (int): Nombre de périodes par an (252 pour jours ouvrables)
    
    Returns:
        float or Series: Rendement moyen arithmétique
    """
    mean_return = returns.mean()
    
    if annualize:
        return mean_return * periods_per_year
    return mean_return

def geometric_mean_return(returns, annualize=True, periods_per_year=252):
    """
    Calcule le rendement moyen géométrique (plus pertinent pour le long terme).
    
    Args:
        returns (Series or DataFrame): Rendements quotidiens
        annualize (bool): Si True, annualise le résultat
        periods_per_year (int): Nombre de périodes par an
    
    Returns:
        float or Series: Rendement moyen géométrique
    """
    if isinstance(returns, pd.DataFrame):
        # Pour un DataFrame (plusieurs actifs)
        geo_means = {}
        for col in returns.columns:
            compound_return = (1 + returns[col]).prod() ** (1 / len(returns)) - 1
            if annualize:
                geo_means[col] = (1 + compound_return) ** periods_per_year - 1
            else:
                geo_means[col] = compound_return
        return pd.Series(geo_means)
    else:
        # Pour une Series (un seul actif)
        compound_return = (1 + returns).prod() ** (1 / len(returns)) - 1
        if annualize:
            return (1 + compound_return) ** periods_per_year - 1
        return compound_return

def cumulative_returns(returns):
    """
    Calcule les rendements cumulatifs.
    
    Args:
        returns (Series or DataFrame): Rendements quotidiens
    
    Returns:
        Series or DataFrame: Rendements cumulatifs
    """
    return (1 + returns).cumprod() - 1

def rolling_returns(returns, window=252):
    """
    Calcule les rendements sur une fenêtre glissante.
    
    Args:
        returns (Series or DataFrame): Rendements quotidiens
        window (int): Taille de la fenêtre en jours
    
    Returns:
        Series or DataFrame: Rendements glissants
    """
    return ((1 + returns).rolling(window).apply(np.prod, raw=True) - 1)
