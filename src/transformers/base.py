from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
import numpy as np

class BaseTransformer(ABC):
    """Base class for data transformers."""
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the generated data."""
        pass

class CorrelationTransformer(BaseTransformer):
    """Transforms data to match target correlations."""
    
    def __init__(self, target_correlations: Dict[tuple, float]):
        self.target_correlations = target_correlations

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Cholesky decomposition to achieve target correlations."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return data
            
        correlation_matrix = np.eye(len(numeric_cols))
        for (field1, field2), target in self.target_correlations.items():
            if field1 in numeric_cols and field2 in numeric_cols:
                i = numeric_cols.get_loc(field1)
                j = numeric_cols.get_loc(field2)
                correlation_matrix[i, j] = target
                correlation_matrix[j, i] = target
                
        L = np.linalg.cholesky(correlation_matrix)
        transformed_data = data.copy()
        transformed_data[numeric_cols] = np.dot(
            data[numeric_cols], L
        )
        
        return transformed_data 