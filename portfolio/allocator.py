"""
Module pour l'allocation d'actifs dans un portefeuille.
"""
import numpy as np
import pandas as pd

def equal_weight_allocation(tickers):
    """
    Crée une allocation à poids égaux.
    
    Args:
        tickers (list): Liste des symboles boursiers
    
    Returns:
        Series: Poids alloués à chaque actif
    """
    n = len(tickers)
    weights = np.array([1/n] * n)
    return pd.Series(weights, index=tickers)

def market_cap_weight_allocation(tickers, market_caps):
    """
    Crée une allocation pondérée par capitalisation boursière.
    
    Args:
        tickers (list): Liste des symboles boursiers
        market_caps (list or Series): Capitalisations boursières correspondantes
    
    Returns:
        Series: Poids alloués à chaque actif
    """
    market_caps = pd.Series(market_caps, index=tickers)
    weights = market_caps / market_caps.sum()
    return weights

def risk_parity_allocation(returns, risk_budget=None, iterations=100):
    """
    Crée une allocation en parité de risque.
    
    Args:
        returns (DataFrame): Rendements quotidiens des actifs
        risk_budget (array, optional): Budget de risque par actif
        iterations (int): Nombre d'itérations
    
    Returns:
        Series: Poids alloués à chaque actif
    """
    n = len(returns.columns)
    
    # Si pas de budget de risque spécifié, utiliser une répartition égale
    if risk_budget is None:
        risk_budget = np.array([1/n] * n)
    
    # Covariance annualisée
    cov = returns.cov() * 252
    
    # Initialisation des poids
    weights = np.array([1/n] * n)
    
    # Algorithme itératif pour la parité de risque
    for _ in range(iterations):
        # Calculer les contributions au risque
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        marginal_contrib = np.dot(cov, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        
        # Calculer les ajustements de poids
        weight_adjustments = risk_budget - risk_contrib / sum(risk_contrib)
        weights += weight_adjustments * 0.01
        
        # Normaliser les poids
        weights = np.maximum(weights, 0)  # Pas de poids négatifs
        weights = weights / sum(weights)
    
    return pd.Series(weights, index=returns.columns)

def rebalance_portfolio(current_weights, target_weights, threshold=0.05):
    """
    Détermine si un rééquilibrage est nécessaire et les transactions requises.
    
    Args:
        current_weights (Series): Poids actuels des actifs
        target_weights (Series): Poids cibles des actifs
        threshold (float): Seuil de tolérance pour le rééquilibrage
    
    Returns:
        dict: Résultat du rééquilibrage avec transactions requises
    """
    # Assurez-vous que les indices correspondent
    common_tickers = current_weights.index.intersection(target_weights.index)
    current = current_weights.loc[common_tickers]
    target = target_weights.loc[common_tickers]
    
    # Calcul des écarts
    diff = target - current
    
    # Vérifier si un rééquilibrage est nécessaire
    need_rebalance = abs(diff).max() > threshold
    
    # Transactions à effectuer (positif = achat, négatif = vente)
    trades = diff if need_rebalance else pd.Series(0, index=common_tickers)
    
    return {
        'need_rebalance': need_rebalance,
        'current_weights': current,
        'target_weights': target,
        'trades': trades
    }
