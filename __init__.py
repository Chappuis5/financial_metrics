"""
Librairie d'analyse de métriques financières pour investissement à long terme.
"""

__version__ = '0.1.0'
from financial_metrics.data import fetchers
from financial_metrics.metrics import returns, risk, ratios
from financial_metrics.portfolio import optimizer, allocator