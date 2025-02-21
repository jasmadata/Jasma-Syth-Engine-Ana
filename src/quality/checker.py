from typing import List, Dict, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class QualityMetrics:
    """Container for data quality metrics."""
    completeness: float
    uniqueness: Dict[str, float]
    value_distribution: Dict[str, Dict[str, float]]
    outliers: Dict[str, List[Any]]

class QualityChecker:
    """Checks quality of generated data."""
    
    def __init__(self, outlier_threshold: float = 3.0):
        self.outlier_threshold = outlier_threshold

    def check_quality(self, data: pd.DataFrame) -> QualityMetrics:
        """Perform comprehensive quality check."""
        return QualityMetrics(
            completeness=self._check_completeness(data),
            uniqueness=self._check_uniqueness(data),
            value_distribution=self._check_distribution(data),
            outliers=self._check_outliers(data)
        )

    def _check_completeness(self, data: pd.DataFrame) -> float:
        """Check for missing values."""
        return 1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])

    def _check_uniqueness(self, data: pd.DataFrame) -> Dict[str, float]:
        """Check uniqueness of values in each column."""
        return {
            col: len(data[col].unique()) / len(data)
            for col in data.columns
        }

    def _check_distribution(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Check value distribution for each column."""
        distributions = {}
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                distributions[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'skewness': data[col].skew(),
                    'kurtosis': data[col].kurtosis()
                }
            else:
                value_counts = data[col].value_counts(normalize=True)
                distributions[col] = value_counts.to_dict()
                
        return distributions

    def _check_outliers(self, data: pd.DataFrame) -> Dict[str, List[Any]]:
        """Detect outliers using Z-score method."""
        outliers = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outliers[col] = data[col][z_scores > self.outlier_threshold].tolist()
            
        return outliers 