"""
Module pour le traitement et la préparation des données financières.
"""
import pandas as pd
import numpy as np

def clean_data(df):
    """
    Nettoie et prépare les données.
    
    Args:
        df (DataFrame): DataFrame de prix bruts
    
    Returns:
        DataFrame: Données nettoyées
    """
    # Supprimer les valeurs manquantes ou les remplacer
    df_clean = df.copy()
    df_clean = df_clean.dropna()
    
    # Vérifier les valeurs aberrantes (method simple Z-score)
    returns = df_clean.pct_change().dropna()
    z_scores = (returns - returns.mean()) / returns.std()
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    
    df_clean = df_clean.loc[filtered_entries.index]
    
    return df_clean

def calculate_returns(df):
    """
    Calcule les rendements quotidiens à partir des prix.
    
    Args:
        df (DataFrame): DataFrame de prix
    
    Returns:
        DataFrame: Rendements quotidiens
    """
    if isinstance(df, pd.Series):
        return df.pct_change().dropna()
    else:
        return df.pct_change().dropna()

def resample_data(df, freq="M"):
    """
    Rééchantillonne les données selon la fréquence désirée.
    
    Args:
        df (DataFrame): DataFrame de prix
        freq (str): Fréquence (D=jour, W=semaine, M=mois, Q=trimestre, Y=année)
    
    Returns:
        DataFrame: Données rééchantillonnées
    """
    return df.resample(freq).last()