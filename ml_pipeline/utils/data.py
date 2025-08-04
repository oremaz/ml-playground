import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Any, Union, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)

def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.1,
    test_size: float = 0.1,
    stratify: Optional[np.ndarray] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, val, and test sets.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    if test_size == 0:
        # Only split into train and val
        val_ratio = val_size
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, stratify=stratify, random_state=random_state
        )
        X_test = np.array([])
        y_test = np.array([])
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify, random_state=random_state
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp if stratify is not None else None, random_state=random_state
        )
    # Ensure all outputs are np.ndarray
    return (
        np.asarray(X_train), np.asarray(X_val), np.asarray(X_test),
        np.asarray(y_train), np.asarray(y_val), np.asarray(y_test)
    )

def get_feature_target_arrays(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract X, y arrays from DataFrame given feature and target columns.
    """
    if len(feature_cols) == 1:
        # If the column contains arrays (e.g., embeddings), convert to list then stack
        arrs = list(df[feature_cols[0]].values)
        X = np.vstack(arrs)
    else:
        X = np.asarray(df[feature_cols].values)
    y = np.asarray(df[target_col].values)
    return X, y

def preprocess_text(text: str) -> str:
    """
    Simple text preprocessing: lowercase, strip, remove extra spaces.
    """
    return ' '.join(text.lower().strip().split())

def _to_list(y: Optional[Union[List[Any], np.ndarray, Any]]) -> List[Any]:
    if y is None:
        return []
    if isinstance(y, np.ndarray):
        return y.tolist()
    if isinstance(y, (list, tuple)):
        return list(y)
    return [y]

class LabelEncoderHelper:
    """
    Helper for label encoding/decoding.
    """
    def __init__(self):
        self.encoder = LabelEncoder()
    def fit(self, y: Optional[Union[List[Any], np.ndarray, Any]]):
        y_list = _to_list(y)
        if not y_list:
            return
        self.encoder.fit(y_list)
    def transform(self, y: Optional[Union[List[Any], np.ndarray, Any]]) -> np.ndarray:
        y_list = _to_list(y)
        if not y_list:
            return np.array([])
        return self.encoder.transform(y_list)
    def inverse_transform(self, y: Union[np.ndarray, List[Any], Any]) -> np.ndarray:
        y_arr = np.asarray(y)
        return self.encoder.inverse_transform(y_arr)
    def classes(self) -> List[Any]:
        return list(np.asarray(self.encoder.classes_))

def load_recipes_data(csv_path: str = "recipes_df.csv") -> pd.DataFrame:
    """Load recipes data from CSV file."""
    return pd.read_csv(csv_path)

def filter_meal_types(df: pd.DataFrame, meal_types: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter DataFrame to keep only specified meal types.
    
    Args:
        df: DataFrame with 'subcategory' column
        meal_types: List of meal types to keep (e.g., ['Lunch recipes', 'Dinner recipes', 'Breakfast recipes'])
    
    Returns:
        Filtered DataFrame
    """
    if meal_types is None:
        meal_types = ['Lunch recipes', 'Dinner recipes', 'Breakfast recipes']
    
    filtered_df = df[df['subcategory'].isin(meal_types)].copy()
    
    # Map to simplified names
    meal_mapping = {
        'Lunch recipes': 'Lunch',
        'Dinner recipes': 'Dinner', 
        'Breakfast recipes': 'Breakfast'
    }
    filtered_df['meal_type'] = filtered_df['subcategory'].map(meal_mapping)
    
    return filtered_df

def compute_imbalance_rate(y: np.ndarray) -> float:
    """
    Compute the imbalance rate for classification tasks.
    
    Args:
        y: Target labels array
    
    Returns:
        Imbalance rate (ratio of majority class count to minority class count)
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) <= 1:
        return 1.0  # No imbalance if only one class
    
    majority_count = float(np.max(counts))
    minority_count = float(np.min(counts))
    
    return majority_count / minority_count

def get_class_distribution(y: np.ndarray) -> Dict:
    """
    Get detailed class distribution information.
    
    Args:
        y: Target labels array
    
    Returns:
        Dictionary with class counts and imbalance information
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    
    distribution = {
        'classes': unique_classes.tolist(),
        'counts': counts.tolist(),
        'total_samples': len(y),
        'num_classes': len(unique_classes),
        'imbalance_rate': compute_imbalance_rate(y),
        'class_ratios': (counts / len(y)).tolist()
    }
    
    return distribution

def prepare_embeddings_data(df: pd.DataFrame, target_column: str, embedding_column: str = 'embeddings_class') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare embeddings and target data for training.
    
    Args:
        df: DataFrame with embeddings and target columns
        target_column: Name of the target column
        embedding_column: Name of the embedding column
    
    Returns:
        Tuple of (X, y) where X is embeddings array and y is target array
    """
    # Convert embeddings from string to numpy array
    if df[embedding_column].dtype == 'object':
        X = np.array([eval(emb) if isinstance(emb, str) else emb for emb in df[embedding_column]])
    else:
        X = df[embedding_column].values
    
    # Prepare target
    y = df[target_column].values
    
    return X, y

def split_data_with_stratification(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data with stratification for classification tasks.
    
    Args:
        X: Features array
        y: Target array
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # For classification tasks, use stratification
    if len(np.unique(y)) > 1:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        # For regression or single-class, use regular split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
