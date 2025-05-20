"""
Module pour l'optimisation de portefeuille.
"""
import numpy as np
import pandas as pd
import scipy.optimize as sco

def portfolio_return(weights, returns):
    """
    Calcule le rendement attendu du portefeuille.
    
    Args:
        weights (array): Poids des actifs
        returns (DataFrame): Rendements moyens des actifs
    
    Returns:
        float: Rendement attendu du portefeuille
    """
    return np.sum(returns.mean() * weights) * 252

def portfolio_volatility(weights, returns):
    """
    Calcule la volatilité du portefeuille.
    
    Args:
        weights (array): Poids des actifs
        returns (DataFrame): Rendements des actifs
    
    Returns:
        float: Volatilité du portefeuille
    """
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

def min_volatility_portfolio(returns):
    """
    Trouve le portefeuille à volatilité minimale.
    
    Args:
        returns (DataFrame): Rendements quotidiens des actifs
    
    Returns:
        dict: Portefeuille optimal avec poids et métriques
    """
    n = len(returns.columns)
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.array([1/n] * n)
    
    result = sco.minimize(portfolio_volatility, initial_weights, args=args,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = pd.Series(result['x'], index=returns.columns)
    
    return {
        'weights': optimal_weights,
        'return': portfolio_return(result['x'], returns),
        'volatility': portfolio_volatility(result['x'], returns),
        'sharpe': portfolio_return(result['x'], returns) / portfolio_volatility(result['x'], returns)
    }

def max_sharpe_portfolio(returns, risk_free_rate=0.0):
    """
    Trouve le portefeuille avec le ratio de Sharpe maximal.
    
    Args:
        returns (DataFrame): Rendements quotidiens des actifs
        risk_free_rate (float): Taux sans risque annualisé
    
    Returns:
        dict: Portefeuille optimal avec poids et métriques
    """
    n = len(returns.columns)
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.array([1/n] * n)
    
    def neg_sharpe_ratio(weights, returns, risk_free_rate):
        p_ret = portfolio_return(weights, returns)
        p_vol = portfolio_volatility(weights, returns)
        return -(p_ret - risk_free_rate) / p_vol
    
    result = sco.minimize(neg_sharpe_ratio, initial_weights, args=args,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = pd.Series(result['x'], index=returns.columns)
    
    return {
        'weights': optimal_weights,
        'return': portfolio_return(result['x'], returns),
        'volatility': portfolio_volatility(result['x'], returns),
        'sharpe': -result['fun']  # Négatif car nous avons minimisé le ratio négatif
    }

def efficient_frontier(returns, target_returns, risk_free_rate=0.0):
    """
    Calcule la frontière efficiente pour une série de rendements cibles.
    
    Args:
        returns (DataFrame): Rendements quotidiens des actifs
        target_returns (array): Liste des rendements cibles
        risk_free_rate (float): Taux sans risque annualisé
    
    Returns:
        DataFrame: Frontière efficiente avec rendements, volatilités et ratios
    """
    n = len(returns.columns)
    
    def portfolio_volatility_for_return(weights, returns, target_return):
        # Fonction à minimiser
        return portfolio_volatility(weights, returns)
    
    results = []
    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda x: portfolio_return(x, returns) - target},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        bounds = tuple((0, 1) for _ in range(n))
        initial_weights = np.array([1/n] * n)
        
        result = sco.minimize(portfolio_volatility_for_return, initial_weights,
                             args=(returns, target), method='SLSQP',
                             bounds=bounds, constraints=constraints)
        
        weights = result['x']
        returns_val = portfolio_return(weights, returns)
        volatility = portfolio_volatility(weights, returns)
        sharpe = (returns_val - risk_free_rate) / volatility
        
        results.append({
            'return': returns_val,
            'volatility': volatility,
            'sharpe': sharpe,
            'weights': pd.Series(weights, index=returns.columns)
        })
        
    return pd.DataFrame(results)

