"""
Advanced Data Cleaning Engine
Handles all cleaning operations with high accuracy
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import re
from datetime import datetime
import logging

from .questionnaire import CleaningPreferences, DataType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Advanced data cleaning engine with ML-powered suggestions"""
    
    def __init__(self, df: pd.DataFrame, preferences: CleaningPreferences):
        self.df = df.copy()
        self.original_df = df.copy()
        self.preferences = preferences
        self.cleaning_log = []
        self.changes_made = {}
    
    def clean(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute complete cleaning pipeline"""
        logger.info("Starting data cleaning process...")
        
        # Step 1: Standardize column names
        if self.preferences.standardize_column_names:
            self._standardize_column_names()
        
        # Step 2: Drop columns with too many missing values
        self._handle_high_missing_columns()
        
        # Step 3: Remove duplicates
        if self.preferences.remove_duplicates:
            self._remove_duplicates()
        
        # Step 4: Handle data type conversions
        if self.preferences.auto_convert_types:
            self._auto_convert_types()
        
        # Step 5: Handle missing values
        self._handle_missing_values()
        
        # Step 6: Detect and handle outliers
        if self.preferences.detect_outliers:
            self._handle_outliers()
        
        # Step 7: Clean text fields
        if self.preferences.clean_text:
            self._clean_text_fields()
        
        # Step 8: Validate special fields
        self._validate_special_fields()
        
        # Step 9: Apply custom rules
        if self.preferences.custom_rules:
            self._apply_custom_rules()
        
        # Generate report
        report = self._generate_report()
        
        logger.info("Data cleaning completed!")
        return self.df, report
    
    def _standardize_column_names(self):
        """Standardize column names according to preferences"""
        old_cols = self.df.columns.tolist()
        new_cols = []
        
        for col in old_cols:
            # Convert to string and clean
            new_col = str(col).strip()
            
            # Remove special characters
            new_col = re.sub(r'[^\w\s]', '_', new_col)
            
            # Apply case style
            if self.preferences.column_name_case == 'snake':
                new_col = re.sub(r'\s+', '_', new_col)
                new_col = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', new_col)
                new_col = new_col.lower()
            elif self.preferences.column_name_case == 'camel':
                words = re.split(r'[\s_]+', new_col)
                new_col = words[0].lower() + ''.join(w.capitalize() for w in words[1:])
            elif self.preferences.column_name_case == 'pascal':
                words = re.split(r'[\s_]+', new_col)
                new_col = ''.join(w.capitalize() for w in words)
            elif self.preferences.column_name_case == 'lower':
                new_col = new_col.lower().replace(' ', '_')
            
            # Remove consecutive underscores
            new_col = re.sub(r'_+', '_', new_col)
            new_col = new_col.strip('_')
            
            # Ensure uniqueness
            if new_col in new_cols:
                counter = 1
                while f"{new_col}_{counter}" in new_cols:
                    counter += 1
                new_col = f"{new_col}_{counter}"
            
            new_cols.append(new_col)
        
        if new_cols != old_cols:
            self.df.columns = new_cols
            self._log_change("Standardized column names", {
                'old_columns': old_cols,
                'new_columns': new_cols
            })
    
    def _handle_high_missing_columns(self):
        """Drop columns with too many missing values"""
        threshold = self.preferences.drop_threshold / 100
        cols_to_drop = []
        
        for col in self.df.columns:
            missing_pct = self.df[col].isnull().sum() / len(self.df)
            if missing_pct > threshold:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self._log_change(f"Dropped {len(cols_to_drop)} columns with >{self.preferences.drop_threshold}% missing", {
                'dropped_columns': cols_to_drop
            })
    
    def _remove_duplicates(self):
        """Remove duplicate rows"""
        initial_count = len(self.df)
        
        subset = self.preferences.duplicate_subset if self.preferences.duplicate_subset else None
        
        self.df = self.df.drop_duplicates(
            subset=subset,
            keep=self.preferences.keep_duplicate
        )
        
        duplicates_removed = initial_count - len(self.df)
        
        if duplicates_removed > 0:
            self._log_change(f"Removed {duplicates_removed} duplicate rows", {
                'duplicates_removed': duplicates_removed,
                'rows_remaining': len(self.df)
            })
    
    def _auto_convert_types(self):
        """Automatically convert columns to appropriate data types"""
        conversions = {}
        
        for col in self.df.columns:
            original_dtype = self.df[col].dtype
            
            # Skip if already appropriate type
            if pd.api.types.is_numeric_dtype(self.df[col]) or \
               pd.api.types.is_datetime64_any_dtype(self.df[col]):
                continue
            
            # Try to convert to numeric
            if self._can_convert_to_numeric(self.df[col]):
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    conversions[col] = f"{original_dtype} -> numeric"
                    continue
                except:
                    pass
            
            # Try to convert to datetime
            if self.preferences.parse_dates and self._can_convert_to_datetime(self.df[col]):
                try:
                    # Convert to datetime with microsecond precision (Arrow-compatible)
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce', 
                                                   format=self._detect_date_format(self.df[col]))
                    # Convert to string to avoid Arrow serialization issues
                    self.df[col] = self.df[col].dt.strftime('%Y-%m-%d %H:%M:%S').replace('NaT', None)
                    conversions[col] = f"{original_dtype} -> datetime (string)"
                    continue
                except:
                    pass
            
            # Try to convert to boolean
            if self._can_convert_to_boolean(self.df[col]):
                try:
                    self.df[col] = self._convert_to_boolean(self.df[col])
                    conversions[col] = f"{original_dtype} -> boolean"
                    continue
                except:
                    pass
            
            # Convert to category if low cardinality
            unique_ratio = self.df[col].nunique() / len(self.df[col].dropna())
            if unique_ratio < 0.05 and self.df[col].nunique() > 1:
                self.df[col] = self.df[col].astype('category')
                conversions[col] = f"{original_dtype} -> category"
        
        if conversions:
            self._log_change(f"Auto-converted {len(conversions)} columns", conversions)
    
    def _can_convert_to_numeric(self, series: pd.Series) -> bool:
        """Check if series can be converted to numeric"""
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        # Try conversion on sample
        try:
            pd.to_numeric(sample, errors='raise')
            return True
        except:
            # Check if at least 90% can be converted
            converted = pd.to_numeric(sample, errors='coerce')
            return converted.notna().sum() / len(sample) > 0.9
    
    def _can_convert_to_datetime(self, series: pd.Series) -> bool:
        """Check if series can be converted to datetime"""
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        # Try multiple date formats
        for fmt in self.preferences.date_formats:
            try:
                pd.to_datetime(sample, format=fmt, errors='raise')
                return True
            except:
                pass
        
        # Try automatic detection
        try:
            converted = pd.to_datetime(sample, errors='coerce')
            return converted.notna().sum() / len(sample) > 0.9
        except:
            return False
    
    def _detect_date_format(self, series: pd.Series) -> Optional[str]:
        """Detect the date format in a series"""
        sample = series.dropna().head(100)
        
        for fmt in self.preferences.date_formats:
            try:
                pd.to_datetime(sample, format=fmt, errors='raise')
                return fmt
            except:
                pass
        
        return None
    
    def _can_convert_to_boolean(self, series: pd.Series) -> bool:
        """Check if series represents boolean values"""
        unique_vals = series.dropna().unique()
        
        if len(unique_vals) > 2:
            return False
        
        bool_values = {
            True, False, 'true', 'false', 'yes', 'no', 
            '1', '0', 1, 0, 'y', 'n', 't', 'f',
            'TRUE', 'FALSE', 'YES', 'NO', 'Y', 'N', 'T', 'F'
        }
        
        return all(v in bool_values for v in unique_vals)
    
    def _convert_to_boolean(self, series: pd.Series) -> pd.Series:
        """Convert series to boolean"""
        true_values = {True, 'true', 'yes', '1', 1, 'y', 't', 'TRUE', 'YES', 'Y', 'T'}
        return series.apply(lambda x: x in true_values if pd.notna(x) else np.nan)
    
    def _handle_missing_values(self):
        """Handle missing values according to preferences"""
        missing_before = self.df.isnull().sum().sum()
        
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue
            
            # Determine fill method based on data type
            if pd.api.types.is_numeric_dtype(self.df[col]):
                method = self.preferences.fill_numeric_method
                
                if method == 'mean':
                    fill_value = self.df[col].mean()
                elif method == 'median':
                    fill_value = self.df[col].median()
                elif method == 'mode':
                    fill_value = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 0
                elif method == 'zero':
                    fill_value = 0
                elif method == 'forward_fill':
                    self.df[col] = self.df[col].fillna(method='ffill')
                    continue
                elif method == 'backward_fill':
                    self.df[col] = self.df[col].fillna(method='bfill')
                    continue
                else:
                    fill_value = self.df[col].median()
                
                self.df[col] = self.df[col].fillna(fill_value)
            
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                # For datetime, use forward fill
                self.df[col] = self.df[col].fillna(method='ffill')
            
            else:
                # Categorical/text
                method = self.preferences.fill_categorical_method
                
                if method == 'mode':
                    fill_value = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                    self.df[col] = self.df[col].fillna(fill_value)
                elif method == 'unknown':
                    self.df[col] = self.df[col].fillna('Unknown')
                elif method == 'forward_fill':
                    self.df[col] = self.df[col].fillna(method='ffill')
        
        missing_after = self.df.isnull().sum().sum()
        
        if missing_before != missing_after:
            self._log_change(f"Filled missing values", {
                'missing_before': int(missing_before),
                'missing_after': int(missing_after),
                'filled': int(missing_before - missing_after)
            })
    
    def _handle_outliers(self):
        """Detect and handle outliers in numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numeric_cols:
            if self.df[col].nunique() < 10:  # Skip low-cardinality numeric columns
                continue
            
            if self.preferences.outlier_method == 'iqr':
                outliers = self._detect_outliers_iqr(self.df[col])
            elif self.preferences.outlier_method == 'zscore':
                outliers = self._detect_outliers_zscore(self.df[col])
            elif self.preferences.outlier_method == 'isolation_forest':
                outliers = self._detect_outliers_isolation_forest(self.df[[col]])
            else:
                continue
            
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                if self.preferences.outlier_action == 'remove':
                    self.df = self.df[~outliers]
                elif self.preferences.outlier_action == 'cap':
                    self._cap_outliers(col, outliers)
                # 'flag' just marks them
                
                outliers_info[col] = int(outlier_count)
        
        if outliers_info:
            self._log_change(f"Handled outliers ({self.preferences.outlier_action})", outliers_info)
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = pd.Series(False, index=series.index)
        outliers.loc[series.notna()] = z_scores > threshold
        return outliers
    
    def _detect_outliers_isolation_forest(self, df: pd.DataFrame, contamination: float = 0.1) -> pd.Series:
        """Detect outliers using Isolation Forest"""
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(df.dropna())
        
        outliers = pd.Series(False, index=df.index)
        outliers.loc[df.dropna().index] = predictions == -1
        return outliers
    
    def _cap_outliers(self, col: str, outliers: pd.Series):
        """Cap outliers at the boundary values"""
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.df.loc[self.df[col] < lower_bound, col] = lower_bound
        self.df.loc[self.df[col] > upper_bound, col] = upper_bound
    
    def _clean_text_fields(self):
        """Clean text fields"""
        text_cols = self.df.select_dtypes(include=['object']).columns
        cleaned_cols = []
        
        for col in text_cols:
            # Skip if datetime or numeric that was loaded as object
            if pd.api.types.is_numeric_dtype(self.df[col]) or \
               pd.api.types.is_datetime64_any_dtype(self.df[col]):
                continue
            
            original_values = self.df[col].copy()
            
            # Strip whitespace
            if self.preferences.strip_whitespace:
                self.df[col] = self.df[col].str.strip()
            
            # Standardize case
            if self.preferences.standardize_case == 'lower':
                self.df[col] = self.df[col].str.lower()
            elif self.preferences.standardize_case == 'upper':
                self.df[col] = self.df[col].str.upper()
            elif self.preferences.standardize_case == 'title':
                self.df[col] = self.df[col].str.title()
            
            # Remove special characters
            if self.preferences.remove_special_chars:
                self.df[col] = self.df[col].str.replace(r'[^\w\s]', '', regex=True)
            
            if not self.df[col].equals(original_values):
                cleaned_cols.append(col)
        
        if cleaned_cols:
            self._log_change(f"Cleaned {len(cleaned_cols)} text columns", {
                'columns': cleaned_cols
            })
    
    def _validate_special_fields(self):
        """Validate email, phone, URL fields"""
        validated = {}
        
        for col in self.df.columns:
            # Email validation
            if self.preferences.validate_emails and 'email' in col.lower():
                pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                valid_mask = self.df[col].str.match(pattern, na=False)
                invalid_count = (~valid_mask).sum()
                if invalid_count > 0:
                    validated[col] = f"{invalid_count} invalid emails"
            
            # Phone validation (basic)
            if self.preferences.validate_phone_numbers and 'phone' in col.lower():
                pattern = r'^[\d\s\-\(\)\+]+$'
                valid_mask = self.df[col].str.match(pattern, na=False)
                invalid_count = (~valid_mask).sum()
                if invalid_count > 0:
                    validated[col] = f"{invalid_count} invalid phone numbers"
            
            # URL validation
            if self.preferences.validate_urls and 'url' in col.lower():
                pattern = r'^https?://[^\s/$.?#].[^\s]*$'
                valid_mask = self.df[col].str.match(pattern, na=False)
                invalid_count = (~valid_mask).sum()
                if invalid_count > 0:
                    validated[col] = f"{invalid_count} invalid URLs"
        
        if validated:
            self._log_change("Validated special fields", validated)
    
    def _apply_custom_rules(self):
        """Apply user-defined custom rules"""
        # This can be extended based on user needs
        for rule_name, rule_config in self.preferences.custom_rules.items():
            # Example: rule_config = {'column': 'age', 'min': 0, 'max': 120}
            pass
    
    def _log_change(self, action: str, details: Any):
        """Log a cleaning action"""
        self.cleaning_log.append({
            'action': action,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"✓ {action}")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive cleaning report"""
        return {
            'summary': {
                'original_shape': self.original_df.shape,
                'final_shape': self.df.shape,
                'rows_removed': len(self.original_df) - len(self.df),
                'columns_removed': len(self.original_df.columns) - len(self.df.columns),
                'memory_saved_mb': (
                    self.original_df.memory_usage(deep=True).sum() - 
                    self.df.memory_usage(deep=True).sum()
                ) / 1024 / 1024
            },
            'data_quality': {
                'missing_values_before': int(self.original_df.isnull().sum().sum()),
                'missing_values_after': int(self.df.isnull().sum().sum()),
                'duplicates_before': int(self.original_df.duplicated().sum()),
                'duplicates_after': int(self.df.duplicated().sum())
            },
            'cleaning_log': self.cleaning_log,
            'column_changes': {
                'original_columns': self.original_df.columns.tolist(),
                'final_columns': self.df.columns.tolist()
            }
        }
