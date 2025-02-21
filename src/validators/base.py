from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
import pandas as pd

class BaseValidator(ABC):
    """Base class for data validators."""
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate the generated data."""
        pass

    @abstractmethod
    def get_validation_report(self) -> Dict[str, Any]:
        """Get detailed validation report."""
        pass

class CorrelationValidator(BaseValidator):
    """Validates correlations between numerical fields."""
    
    def __init__(self, target_correlations: Dict[tuple, float], tolerance: float = 0.1):
        self.target_correlations = target_correlations
        self.tolerance = tolerance
        self.validation_report = {}

    def validate(self, data: pd.DataFrame) -> bool:
        corr_matrix = data.corr()
        valid = True
        
        for (field1, field2), target in self.target_correlations.items():
            if field1 in data.columns and field2 in data.columns:
                actual = corr_matrix.loc[field1, field2]
                diff = abs(actual - target)
                valid &= diff <= self.tolerance
                
                self.validation_report[(field1, field2)] = {
                    'target': target,
                    'actual': actual,
                    'difference': diff,
                    'valid': diff <= self.tolerance
                }
                
        return valid

    def get_validation_report(self) -> Dict[str, Any]:
        return self.validation_report 