"""
Module pour le calcul des mesures de risque.
"""
import numpy as np
import pandas as pd

def volatility(returns, annualize=True, periods_per_year=252):
    """
    Calcule la volatilité (écart-type des rendements).
    
    Args:
        returns (Series or DataFrame): Rendements quotidiens
        annualize (bool): Si True, annualise le résultat
        periods_per_year (int): Nombre de périodes par an
    
    Returns:
        float or Series: Volatilité
    """
    vol = returns.std()
    if annualize:
        return vol * np.sqrt(periods_per_year)
    return vol

def max_drawdown(price_data):
    """
    Calcule le maximum drawdown.
    
    Args:
        price_data (Series or DataFrame): Série de prix
    
    Returns:
        float or Series: Maximum drawdown en pourcentage
    """
    if isinstance(price_data, pd.DataFrame):
        result = {}
        for col in price_data.columns:
            roll_max = price_data[col].cummax()
            drawdown = price_data[col] / roll_max - 1
            result[col] = drawdown.min()
        return pd.Series(result)
    else:
        roll_max = price_data.cummax()
        drawdown = price_data / roll_max - 1
        return drawdown.min()

def drawdown_periods(price_data):
    """
    Identifie les périodes de drawdown.
    
    Args:
        price_data (Series): Série de prix
    
    Returns:
        DataFrame: Périodes de drawdown avec début, fin et profondeur
    """
    roll_max = price_data.cummax()
    drawdown = price_data / roll_max - 1
    
    # Identifier les périodes
    is_dd = drawdown < 0
    
    # Grouper par périodes consécutives
    dd_groups = (is_dd != is_dd.shift()).cumsum()[is_dd]
    
    # Résultat
    result = []
    
    for group_id, group_indices in dd_groups.groupby(dd_groups):
        if len(group_indices) > 0:
            start_date = group_indices.index[0]
            end_date = group_indices.index[-1]
            depth = drawdown.loc[start_date:end_date].min()
            recovery_date = None
            
            # Chercher la date de récupération
            if end_date != price_data.index[-1]:
                recovery_dates = price_data[end_date:].loc[price_data >= roll_max[end_date]].index
                if len(recovery_dates) > 0:
                    recovery_date = recovery_dates[0]
            
            result.append({
                'start': start_date,
                'end': end_date,
                'depth': depth,
                'recovery': recovery_date,
                'duration': len(group_indices)
            })
            
    return pd.DataFrame(result)

def downside_risk(returns, min_acceptable_return=0, annualize=True, periods_per_year=252):
    """
    Calcule le risque à la baisse (deviation semi-standard).
    
    Args:
        returns (Series or DataFrame): Rendements quotidiens
        min_acceptable_return (float): Rendement minimum acceptable
        annualize (bool): Si True, annualise le résultat
        periods_per_year (int): Nombre de périodes par an
    
    Returns:
        float or Series: Risque à la baisse
    """
    downside_diff = returns.copy()
    if isinstance(returns, pd.DataFrame):
        for col in returns.columns:
            downside_diff[col] = np.minimum(returns[col] - min_acceptable_return, 0)
    else:
        downside_diff = np.minimum(returns - min_acceptable_return, 0)
    
    downside_diff = downside_diff ** 2
    downside_risk = np.sqrt(downside_diff.mean())
    
    if annualize:
        return downside_risk * np.sqrt(periods_per_year)
    return downside_risk

def value_at_risk(returns, confidence_level=0.05):
    """
    Calcule la Value at Risk (non paramétrique).
    
    Args:
        returns (Series or DataFrame): Rendements quotidiens
        confidence_level (float): Niveau de confiance (5% = 0.05)
    
    Returns:
        float or Series: Value at Risk
    """
    return returns.quantile(confidence_level)
