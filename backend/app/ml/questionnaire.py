"""
Interactive Questionnaire System for Data Cleaning
Asks intelligent questions before cleaning to understand data context
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np


class CleaningStrategy(Enum):
    """Available cleaning strategies"""
    CONSERVATIVE = "conservative"  # Minimal changes, preserve data
    MODERATE = "moderate"  # Balanced approach
    AGGRESSIVE = "aggressive"  # Maximum cleaning, remove ambiguous data


class DataType(Enum):
    """Common data types"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    MIXED = "mixed"


@dataclass
class ColumnProfile:
    """Profile of a single column"""
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[Any]
    inferred_type: DataType
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class CleaningPreferences:
    """User preferences for data cleaning"""
    strategy: CleaningStrategy = CleaningStrategy.MODERATE
    
    # Missing value handling
    drop_threshold: float = 0.7  # Drop columns with >70% missing
    fill_numeric_method: str = "median"  # mean, median, mode, forward_fill, backward_fill
    fill_categorical_method: str = "mode"  # mode, unknown, forward_fill
    
    # Duplicate handling
    remove_duplicates: bool = True
    duplicate_subset: Optional[List[str]] = None
    keep_duplicate: str = "first"  # first, last, False (remove all)
    
    # Outlier handling
    detect_outliers: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_action: str = "flag"  # flag, remove, cap
    
    # Data type corrections
    auto_convert_types: bool = True
    parse_dates: bool = True
    date_formats: List[str] = field(default_factory=lambda: ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"])
    
    # Text cleaning
    clean_text: bool = True
    strip_whitespace: bool = True
    standardize_case: Optional[str] = None  # None, "lower", "upper", "title"
    remove_special_chars: bool = False
    
    # Standardization
    standardize_column_names: bool = True
    column_name_case: str = "snake"  # snake, camel, pascal, lower
    
    # Validation
    validate_emails: bool = False
    validate_phone_numbers: bool = False
    validate_urls: bool = False
    
    # Custom rules
    custom_rules: Dict[str, Any] = field(default_factory=dict)


class CleaningQuestionnaire:
    """Interactive questionnaire for determining cleaning preferences"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.profile = self._profile_data()
        self.preferences = CleaningPreferences()
    
    def _profile_data(self) -> Dict[str, ColumnProfile]:
        """Profile each column in the dataset"""
        profiles = {}
        
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / len(self.df)) * 100
            unique_count = self.df[col].nunique()
            unique_pct = (unique_count / len(self.df)) * 100
            
            # Get sample non-null values
            sample_values = self.df[col].dropna().head(5).tolist()
            
            # Infer semantic type
            inferred_type = self._infer_column_type(self.df[col])
            
            # Generate suggested actions
            suggestions = self._suggest_actions(col, null_pct, unique_count, inferred_type)
            
            profiles[col] = ColumnProfile(
                name=col,
                dtype=dtype,
                null_count=null_count,
                null_percentage=null_pct,
                unique_count=unique_count,
                unique_percentage=unique_pct,
                sample_values=sample_values,
                inferred_type=inferred_type,
                suggested_actions=suggestions
            )
        
        return profiles
    
    def _infer_column_type(self, series: pd.Series) -> DataType:
        """Infer the semantic type of a column"""
        # Drop nulls for analysis
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return DataType.MIXED
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            return DataType.NUMERIC
        
        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATETIME
        
        # Check if boolean
        unique_vals = non_null.unique()
        if len(unique_vals) <= 2:
            bool_values = {True, False, 'true', 'false', 'yes', 'no', '1', '0', 1, 0, 'y', 'n'}
            if all(str(v).lower() in bool_values for v in unique_vals):
                return DataType.BOOLEAN
        
        # Check if categorical (low cardinality)
        if len(unique_vals) / len(non_null) < 0.05:  # Less than 5% unique
            return DataType.CATEGORICAL
        
        # Try to parse as date
        try:
            pd.to_datetime(non_null.head(100), errors='coerce')
            if non_null.head(100).apply(lambda x: pd.to_datetime(x, errors='coerce')).notna().sum() > 50:
                return DataType.DATETIME
        except:
            pass
        
        # Default to text
        return DataType.TEXT
    
    def _suggest_actions(self, col_name: str, null_pct: float, 
                        unique_count: int, data_type: DataType) -> List[str]:
        """Suggest cleaning actions for a column"""
        suggestions = []
        
        if null_pct > 70:
            suggestions.append(f"⚠️ Consider dropping (>{null_pct:.1f}% missing)")
        elif null_pct > 30:
            suggestions.append(f"⚠️ High missing rate ({null_pct:.1f}%)")
        elif null_pct > 0:
            suggestions.append(f"Fill missing values ({null_pct:.1f}% missing)")
        
        if unique_count == 1:
            suggestions.append("⚠️ Constant column - consider dropping")
        
        if data_type == DataType.DATETIME:
            suggestions.append("Parse as datetime")
        elif data_type == DataType.BOOLEAN:
            suggestions.append("Convert to boolean")
        elif data_type == DataType.NUMERIC:
            suggestions.append("Check for outliers")
        elif data_type == DataType.CATEGORICAL:
            suggestions.append("Consider encoding for ML")
        
        return suggestions
    
    def generate_questions(self) -> List[Dict[str, Any]]:
        """Generate contextual questions based on data profile"""
        questions = []
        
        # General strategy
        questions.append({
            'id': 'strategy',
            'type': 'choice',
            'question': 'What cleaning strategy do you prefer?',
            'options': [
                ('conservative', 'Conservative - Keep as much data as possible'),
                ('moderate', 'Moderate - Balanced approach (Recommended)'),
                ('aggressive', 'Aggressive - Remove ambiguous/problematic data')
            ],
            'default': 'moderate'
        })
        
        # Missing values
        high_missing_cols = [col for col, prof in self.profile.items() 
                           if prof.null_percentage > 30]
        
        if high_missing_cols:
            questions.append({
                'id': 'drop_threshold',
                'type': 'number',
                'question': f'Drop columns with what % of missing values? (Found {len(high_missing_cols)} columns with >30% missing)',
                'default': 70,
                'min': 0,
                'max': 100
            })
        
        # Numeric columns
        numeric_cols = [col for col, prof in self.profile.items() 
                       if prof.inferred_type == DataType.NUMERIC]
        
        if numeric_cols:
            questions.append({
                'id': 'fill_numeric_method',
                'type': 'choice',
                'question': f'How to fill missing values in numeric columns? ({len(numeric_cols)} found)',
                'options': [
                    ('median', 'Median (robust to outliers)'),
                    ('mean', 'Mean'),
                    ('mode', 'Mode (most frequent)'),
                    ('forward_fill', 'Forward fill'),
                    ('zero', 'Fill with 0')
                ],
                'default': 'median'
            })
            
            questions.append({
                'id': 'detect_outliers',
                'type': 'boolean',
                'question': 'Detect and handle outliers in numeric columns?',
                'default': True
            })
        
        # Categorical columns
        categorical_cols = [col for col, prof in self.profile.items() 
                          if prof.inferred_type == DataType.CATEGORICAL]
        
        if categorical_cols:
            questions.append({
                'id': 'fill_categorical_method',
                'type': 'choice',
                'question': f'How to fill missing values in categorical columns? ({len(categorical_cols)} found)',
                'options': [
                    ('mode', 'Mode (most frequent)'),
                    ('unknown', 'Fill with "Unknown"'),
                    ('forward_fill', 'Forward fill')
                ],
                'default': 'mode'
            })
        
        # Datetime columns
        datetime_cols = [col for col, prof in self.profile.items() 
                        if prof.inferred_type == DataType.DATETIME]
        
        if datetime_cols:
            questions.append({
                'id': 'parse_dates',
                'type': 'boolean',
                'question': f'Automatically parse date columns? ({len(datetime_cols)} detected)',
                'default': True
            })
        
        # Duplicates
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            questions.append({
                'id': 'remove_duplicates',
                'type': 'boolean',
                'question': f'Remove duplicate rows? (Found {dup_count} duplicates)',
                'default': True
            })
        
        # Text cleaning
        text_cols = [col for col, prof in self.profile.items() 
                    if prof.inferred_type == DataType.TEXT]
        
        if text_cols:
            questions.append({
                'id': 'clean_text',
                'type': 'boolean',
                'question': f'Clean text columns (trim whitespace, etc.)? ({len(text_cols)} text columns)',
                'default': True
            })
        
        # Column names
        has_special_chars = any(not col.replace('_', '').replace(' ', '').isalnum() 
                               for col in self.df.columns)
        
        if has_special_chars:
            questions.append({
                'id': 'standardize_column_names',
                'type': 'boolean',
                'question': 'Standardize column names (snake_case, remove special characters)?',
                'default': True
            })
        
        return questions
    
    def apply_answers(self, answers: Dict[str, Any]) -> CleaningPreferences:
        """Apply user answers to preferences"""
        for key, value in answers.items():
            if hasattr(self.preferences, key):
                if key == 'strategy' and isinstance(value, str):
                    self.preferences.strategy = CleaningStrategy(value)
                else:
                    setattr(self.preferences, key, value)
        
        return self.preferences
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the data profile"""
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'total_missing_cells': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'duplicate_rows': self.df.duplicated().sum(),
            'column_types': {
                'numeric': len([p for p in self.profile.values() if p.inferred_type == DataType.NUMERIC]),
                'categorical': len([p for p in self.profile.values() if p.inferred_type == DataType.CATEGORICAL]),
                'datetime': len([p for p in self.profile.values() if p.inferred_type == DataType.DATETIME]),
                'text': len([p for p in self.profile.values() if p.inferred_type == DataType.TEXT]),
                'boolean': len([p for p in self.profile.values() if p.inferred_type == DataType.BOOLEAN])
            },
            'columns': {col: {
                'type': prof.inferred_type.value,
                'missing': f"{prof.null_percentage:.1f}%",
                'unique': prof.unique_count,
                'suggestions': prof.suggested_actions
            } for col, prof in self.profile.items()}
        }
