import argparse
import logging
from typing import Dict, Any, List
import random
import string
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pathlib import Path
import sys
from faker import Faker
from src.validators.base import BaseValidator
from src.transformers.base import BaseTransformer
from src.quality.checker import QualityChecker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#base abstract generator class
class BaseGenerator(ABC):
    """Base abstract generator class for all data generators."""
    
    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """Generate n samples of data.
        
        Args:
            n (int): Number of samples to generate
            
        Returns:
            np.ndarray: Generated data
        """
        pass

    def validate_params(self) -> None:
        """Validate the generator parameters."""
        pass

#numerical data generator
class NumericalGenerator(BaseGenerator):
    """Generator for numerical data following various distributions."""
    
    SUPPORTED_DISTRIBUTIONS = ['normal', 'uniform', 'lognormal', 'exponential']
    
    def __init__(
        self,
        mean: float = 0,
        std_dev: float = 1,
        min_value: float = None,
        max_value: float = None,
        distribution: str = 'normal',
        **kwargs
    ):
        self.mean = mean
        self.std_dev = std_dev
        self.min_value = min_value
        self.max_value = max_value
        self.distribution = distribution
        self.validate_params()

    def validate_params(self) -> None:
        """Validate numerical parameters."""
        if self.distribution not in self.SUPPORTED_DISTRIBUTIONS:
            raise ValueError(f"Distribution must be one of {self.SUPPORTED_DISTRIBUTIONS}")
            
        if self.min_value is not None and self.max_value is not None:
            if self.min_value >= self.max_value:
                raise ValueError("min_value must be less than max_value")
        if self.std_dev <= 0:
            raise ValueError("std_dev must be positive")

    def generate(self, n: int) -> np.ndarray:
        """Generate n samples from the specified distribution."""
        if n <= 0:
            raise ValueError("Number of samples must be positive")
            
        # Generate data based on distribution type
        if self.distribution == 'normal':
            data = np.random.normal(self.mean, self.std_dev, n)
        elif self.distribution == 'uniform':
            data = np.random.uniform(
                self.min_value or self.mean - self.std_dev,
                self.max_value or self.mean + self.std_dev,
                n
            )
        elif self.distribution == 'lognormal':
            data = np.random.lognormal(np.log(self.mean), self.std_dev, n)
        elif self.distribution == 'exponential':
            data = np.random.exponential(self.mean, n)
        
        # Apply bounds if specified
        if self.min_value is not None:
            data = np.maximum(data, self.min_value)
        if self.max_value is not None:
            data = np.minimum(data, self.max_value)
            
        return data

#categorical data generator
class CategoricalGenerator(BaseGenerator):
    """Generator for categorical data."""
    
    def __init__(self, categories: list, probabilities: list = None):
        self.categories = categories
        self.probabilities = probabilities
        self.validate_params()

    def validate_params(self) -> None:
        """Validate categorical parameters."""
        if not self.categories:
            raise ValueError("Categories list cannot be empty")
        if self.probabilities:
            if len(self.categories) != len(self.probabilities):
                raise ValueError("Categories and probabilities must have same length")
            if not np.isclose(sum(self.probabilities), 1.0):
                raise ValueError("Probabilities must sum to 1")

    def generate(self, n: int) -> list:
        return random.choices(self.categories, weights=self.probabilities, k=n)

#time-series data generator
class TimeSeriesGenerator(BaseGenerator):
    """Generator for time series data with patterns."""
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        frequency: str = '1D',
        distribution: str = 'uniform',
        peak_hours: list = None,
        **kwargs
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.frequency = frequency
        self.distribution = distribution
        self.peak_hours = peak_hours or []
        self.validate_params()

    def validate_params(self) -> None:
        """Validate time series parameters."""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        if self.distribution not in ['uniform', 'normal']:
            raise ValueError("distribution must be either 'uniform' or 'normal'")

    def generate(self, n: int) -> np.ndarray:
        """Generate time series data with optional peak hour patterns."""
        date_range = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=self.frequency
        )
        
        if self.distribution == 'uniform':
            dates = np.random.choice(date_range, size=n, replace=False)
        elif self.distribution == 'normal':
            # Generate with preference for peak hours
            weights = np.ones(len(date_range))
            for hour in self.peak_hours:
                mask = date_range.hour == hour
                weights[mask] = 3.0  # Triple probability for peak hours
            weights = weights / weights.sum()
            dates = np.random.choice(date_range, size=n, p=weights)
        
        return np.sort(dates)

#DataSynthesizer to manage and synthesize datasets
class DataSynthesizer:
    """Main class for synthesizing data based on configuration."""
    
    def __init__(self, config_path: str = None):
        self.generators: Dict[str, BaseGenerator] = {}
        self.validators: List[BaseValidator] = []
        self.transformers: List[BaseTransformer] = []
        self.quality_checker = QualityChecker()
        
        self.generator_factories = {
            'numerical': self._create_numerical_generator,
            'categorical': self._create_categorical_generator,
            'timeseries': self._create_timeseries_generator,
            'address': lambda params: AddressGenerator(**params),
            'phone': lambda params: PhoneGenerator(**params),
            'email': lambda params: EmailGenerator(**params),
            'ip': lambda params: IPAddressGenerator(**params),
            'color': lambda params: ColorGenerator(**params)
        }
        
        if config_path:
            self.load_config(config_path)

    def _create_numerical_generator(self, params: Dict[str, Any]) -> NumericalGenerator:
        return NumericalGenerator(**params)

    def _create_categorical_generator(self, params: Dict[str, Any]) -> CategoricalGenerator:
        return CategoricalGenerator(**params)

    def _create_timeseries_generator(self, params: Dict[str, Any]) -> TimeSeriesGenerator:
        return TimeSeriesGenerator(**params)

    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            for field_name, field_config in config['generators'].items():
                generator_type = field_config['type']
                generator_params = field_config['params']
                
                if generator_type not in self.generator_factories:
                    raise ValueError(f"Unknown generator type: {generator_type}")
                
                generator = self.generator_factories[generator_type](generator_params)
                self.add_generator(field_name, generator)
                
            logger.info(f"Successfully loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def add_generator(self, name: str, generator: BaseGenerator) -> None:
        """Add a new generator to the synthesizer."""
        if not isinstance(generator, BaseGenerator):
            raise ValueError(f"Generator for {name} must be an instance of BaseGenerator")
        self.generators[name] = generator
        logger.debug(f"Added generator: {name}")

    def add_validator(self, validator: BaseValidator) -> None:
        """Add a data validator."""
        self.validators.append(validator)

    def add_transformer(self, transformer: BaseTransformer) -> None:
        """Add a data transformer."""
        self.transformers.append(transformer)

    def generate_data(self, n: int) -> pd.DataFrame:
        """Generate and validate synthetic data."""
        try:
            # Generate raw data
            synthesized_data = {}
            for name, generator in self.generators.items():
                synthesized_data[name] = generator.generate(n)
            data = pd.DataFrame(synthesized_data)
            
            # Apply transformations
            for transformer in self.transformers:
                data = transformer.transform(data)
            
            # Validate data
            for validator in self.validators:
                if not validator.validate(data):
                    logger.warning(
                        f"Validation failed: {validator.get_validation_report()}"
                    )
            
            # Check quality
            quality_metrics = self.quality_checker.check_quality(data)
            logger.info(f"Data quality metrics: {quality_metrics}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating data: {str(e)}")
            raise

def save_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to CSV file."""
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate synthetic data based on configuration.')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--samples', type=int, default=100,
                      help='Number of samples to generate (default: 100)')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Output directory for generated data (default: output)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.samples <= 0:
        parser.error("Number of samples must be positive")
    
    return args

#testing
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Initialize synthesizer with fallback to default config
        synthesizer = DataSynthesizer(args.config)
        
        # Generate data
        logger.info(f"Generating {args.samples} samples...")
        synthetic_data = synthesizer.generate_data(args.samples)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f"synthetic_data_{timestamp}.csv"
        save_to_csv(synthetic_data, output_path)
        
        # Display sample
        print("\nFirst few rows of generated data:")
        print(synthetic_data.head())
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)  # Exit with error code
# Add these new generator classes

class AddressGenerator(BaseGenerator):
    """Generator for realistic address data."""
    
    def __init__(self, country='US', locale='en_US', **kwargs):
        self.faker = Faker(locale)
        self.country = country
        super().__init__(**kwargs)

    def generate(self, n: int) -> list:
        return [self.faker.address() for _ in range(n)]

class PhoneGenerator(BaseGenerator):
    """Generator for phone numbers."""
    
    def __init__(self, format='###-###-####', **kwargs):
        self.format = format
        super().__init__(**kwargs)

    def generate(self, n: int) -> list:
        return [''.join(str(random.randint(0, 9)) if c == '#' else c 
                for c in self.format) for _ in range(n)]

class EmailGenerator(BaseGenerator):
    """Generator for email addresses."""
    
    def __init__(self, domains=None, **kwargs):
        self.domains = domains or ['gmail.com', 'yahoo.com', 'hotmail.com']
        super().__init__(**kwargs)

    def generate(self, n: int) -> list:
        emails = []
        for _ in range(n):
            name = ''.join(random.choices(string.ascii_lowercase, k=8))
            domain = random.choice(self.domains)
            emails.append(f"{name}@{domain}")
        return emails

class IPAddressGenerator(BaseGenerator):
    """Generator for IP addresses."""
    
    def generate(self, n: int) -> list:
        return ['.'.join(str(random.randint(0, 255)) for _ in range(4))
                for _ in range(n)]

class ColorGenerator(BaseGenerator):
    """Generator for color codes."""
    
    def __init__(self, format='hex', **kwargs):
        self.format = format
        super().__init__(**kwargs)

    def generate(self, n: int) -> list:
        if self.format == 'hex':
            return ['#' + ''.join(random.choices('0123456789ABCDEF', k=6))
                    for _ in range(n)]
        else:
            return [tuple(random.randint(0, 255) for _ in range(3))
                    for _ in range(n)]

