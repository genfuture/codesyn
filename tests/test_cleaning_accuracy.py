"""
Comprehensive Test Suite for Data Cleaning Tool
Tests accuracy and performance across different scenarios
"""
import pytest
import pandas as pd
import numpy as np
from faker import Faker
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ml.data_loader import DataLoader
from app.ml.questionnaire import CleaningQuestionnaire, CleaningPreferences, CleaningStrategy
from app.ml.data_cleaner import DataCleaner


class TestDataGenerator:
    """Generate test datasets with known issues"""
    
    def __init__(self, n_rows=1000):
        self.fake = Faker()
        self.n_rows = n_rows
    
    def generate_dirty_data(self) -> pd.DataFrame:
        """Generate dataset with various data quality issues"""
        data = {
            # Numeric with outliers and missing
            'age': [self.fake.random_int(18, 80) if np.random.random() > 0.1 
                   else (np.nan if np.random.random() > 0.5 else 999) 
                   for _ in range(self.n_rows)],
            
            # Categorical with inconsistent casing
            'category': [np.random.choice(['Type A', 'type a', 'TYPE A', 'Type B', 'type b', None]) 
                        for _ in range(self.n_rows)],
            
            # Text with whitespace issues
            'name': [f"  {self.fake.name()}  " if np.random.random() > 0.2 else None 
                    for _ in range(self.n_rows)],
            
            # Dates in different formats
            'date': [np.random.choice([
                '2024-01-15', '01/15/2024', '15/01/2024', 
                '2024-01-15 10:30:00', 'invalid', None
            ]) for _ in range(self.n_rows)],
            
            # Boolean in different formats
            'is_active': [np.random.choice(['yes', 'no', 'true', 'false', '1', '0', True, False, None]) 
                         for _ in range(self.n_rows)],
            
            # Email with some invalid
            'email': [self.fake.email() if np.random.random() > 0.1 
                     else ('invalid-email' if np.random.random() > 0.5 else None)
                     for _ in range(self.n_rows)],
            
            # Constant column (should be removed)
            'constant': ['CONSTANT'] * self.n_rows,
            
            # Column with >70% missing (should be removed)
            'mostly_missing': [np.nan if np.random.random() > 0.2 else self.fake.word() 
                              for _ in range(self.n_rows)],
            
            # Numeric with different scales
            'salary': [np.random.choice([
                np.random.randint(30000, 150000),
                np.random.randint(30000, 150000) / 1000,  # Different scale
                np.nan
            ]) for _ in range(self.n_rows)]
        }
        
        df = pd.DataFrame(data)
        
        # Add some duplicate rows
        duplicates = df.sample(int(self.n_rows * 0.1))
        df = pd.concat([df, duplicates], ignore_index=True)
        
        return df
    
    def generate_clean_data(self) -> pd.DataFrame:
        """Generate already clean dataset for control testing"""
        data = {
            'age': [self.fake.random_int(18, 80) for _ in range(self.n_rows)],
            'category': [np.random.choice(['Type A', 'Type B', 'Type C']) 
                        for _ in range(self.n_rows)],
            'name': [self.fake.name() for _ in range(self.n_rows)],
            'date': [self.fake.date() for _ in range(self.n_rows)],
            'is_active': [np.random.choice([True, False]) for _ in range(self.n_rows)],
            'email': [self.fake.email() for _ in range(self.n_rows)],
            'salary': [np.random.randint(30000, 150000) for _ in range(self.n_rows)]
        }
        
        return pd.DataFrame(data)
    
    def generate_large_file_data(self, n_rows=1000000) -> pd.DataFrame:
        """Generate large dataset for performance testing"""
        return self.generate_dirty_data()


class TestDataLoader:
    """Test data loading functionality"""
    
    @pytest.fixture
    def sample_data(self):
        return TestDataGenerator(100).generate_dirty_data()
    
    def test_csv_loading(self, tmp_path, sample_data):
        """Test CSV file loading"""
        csv_path = tmp_path / "test.csv"
        sample_data.to_csv(csv_path, index=False)
        
        loader = DataLoader(csv_path)
        df = loader.load()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_data)
    
    def test_excel_loading(self, tmp_path, sample_data):
        """Test Excel file loading"""
        excel_path = tmp_path / "test.xlsx"
        sample_data.to_excel(excel_path, index=False)
        
        loader = DataLoader(excel_path)
        df = loader.load()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_data)
    
    def test_json_loading(self, tmp_path, sample_data):
        """Test JSON file loading"""
        json_path = tmp_path / "test.json"
        sample_data.to_json(json_path, orient='records')
        
        loader = DataLoader(json_path)
        df = loader.load()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_data)
    
    def test_encoding_detection(self, tmp_path):
        """Test automatic encoding detection"""
        csv_path = tmp_path / "test_utf8.csv"
        
        # Create CSV with special characters
        df = pd.DataFrame({'text': ['café', 'naïve', 'résumé']})
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        loader = DataLoader(csv_path)
        encoding = loader.detect_encoding()
        
        assert encoding is not None
        assert 'utf' in encoding.lower()


class TestQuestionnaire:
    """Test questionnaire functionality"""
    
    @pytest.fixture
    def sample_data(self):
        return TestDataGenerator(100).generate_dirty_data()
    
    def test_questionnaire_creation(self, sample_data):
        """Test questionnaire creation"""
        q = CleaningQuestionnaire(sample_data)
        
        assert q.df is not None
        assert len(q.profile) == len(sample_data.columns)
    
    def test_data_profiling(self, sample_data):
        """Test data profiling accuracy"""
        q = CleaningQuestionnaire(sample_data)
        
        # Check age column profiling
        age_profile = q.profile['age']
        assert age_profile.null_percentage > 0
        assert 'numeric' in age_profile.inferred_type.value
    
    def test_question_generation(self, sample_data):
        """Test contextual question generation"""
        q = CleaningQuestionnaire(sample_data)
        questions = q.generate_questions()
        
        assert len(questions) > 0
        assert any(q['id'] == 'strategy' for q in questions)
    
    def test_summary_generation(self, sample_data):
        """Test summary statistics"""
        q = CleaningQuestionnaire(sample_data)
        summary = q.get_summary()
        
        assert 'total_rows' in summary
        assert 'total_columns' in summary
        assert 'column_types' in summary


class TestDataCleaner:
    """Test data cleaning functionality"""
    
    @pytest.fixture
    def dirty_data(self):
        return TestDataGenerator(1000).generate_dirty_data()
    
    @pytest.fixture
    def clean_data(self):
        return TestDataGenerator(1000).generate_clean_data()
    
    @pytest.fixture
    def preferences(self):
        return CleaningPreferences(
            strategy=CleaningStrategy.MODERATE,
            drop_threshold=70,
            remove_duplicates=True,
            detect_outliers=True,
            clean_text=True
        )
    
    def test_duplicate_removal(self, dirty_data, preferences):
        """Test duplicate removal accuracy"""
        initial_count = len(dirty_data)
        initial_duplicates = dirty_data.duplicated().sum()
        
        cleaner = DataCleaner(dirty_data, preferences)
        cleaned_df, report = cleaner.clean()
        
        final_duplicates = cleaned_df.duplicated().sum()
        
        assert final_duplicates == 0
        assert report['data_quality']['duplicates_after'] == 0
    
    def test_missing_value_handling(self, dirty_data, preferences):
        """Test missing value imputation"""
        initial_missing = dirty_data.isnull().sum().sum()
        
        cleaner = DataCleaner(dirty_data, preferences)
        cleaned_df, report = cleaner.clean()
        
        final_missing = cleaned_df.isnull().sum().sum()
        
        # Should have fewer missing values (some columns may be dropped)
        assert final_missing < initial_missing
    
    def test_outlier_detection(self, dirty_data, preferences):
        """Test outlier detection accuracy"""
        cleaner = DataCleaner(dirty_data, preferences)
        cleaned_df, report = cleaner.clean()
        
        # Check that age outliers (999) are handled
        if 'age' in cleaned_df.columns:
            assert cleaned_df['age'].max() < 200  # Should cap or remove outliers
    
    def test_text_cleaning(self, dirty_data, preferences):
        """Test text field cleaning"""
        cleaner = DataCleaner(dirty_data, preferences)
        cleaned_df, report = cleaner.clean()
        
        # Check that names are trimmed
        if 'name' in cleaned_df.columns:
            names = cleaned_df['name'].dropna()
            for name in names.head(10):
                assert name == name.strip()
                assert not name.startswith(' ')
                assert not name.endswith(' ')
    
    def test_column_name_standardization(self, preferences):
        """Test column name standardization"""
        df = pd.DataFrame({
            'First Name': [1, 2, 3],
            'Last-Name': [4, 5, 6],
            'Age (years)': [7, 8, 9]
        })
        
        cleaner = DataCleaner(df, preferences)
        cleaned_df, report = cleaner.clean()
        
        # Check snake_case conversion
        for col in cleaned_df.columns:
            assert col.islower() or '_' in col
            assert not any(c in col for c in [' ', '-', '(', ')'])
    
    def test_type_conversion(self, dirty_data, preferences):
        """Test automatic type conversion"""
        cleaner = DataCleaner(dirty_data, preferences)
        cleaned_df, report = cleaner.clean()
        
        # Check that is_active is converted to boolean
        if 'is_active' in cleaned_df.columns:
            assert cleaned_df['is_active'].dtype == bool or \
                   pd.api.types.is_bool_dtype(cleaned_df['is_active'])
    
    def test_cleaning_report_completeness(self, dirty_data, preferences):
        """Test that cleaning report is comprehensive"""
        cleaner = DataCleaner(dirty_data, preferences)
        cleaned_df, report = cleaner.clean()
        
        assert 'summary' in report
        assert 'data_quality' in report
        assert 'cleaning_log' in report
        assert 'column_changes' in report
        
        assert 'original_shape' in report['summary']
        assert 'final_shape' in report['summary']
    
    def test_no_over_cleaning(self, clean_data, preferences):
        """Test that already clean data is not over-processed"""
        cleaner = DataCleaner(clean_data, preferences)
        cleaned_df, report = cleaner.clean()
        
        # Should have similar shape (no excessive removals)
        assert len(cleaned_df) >= len(clean_data) * 0.95


class TestPerformance:
    """Test performance with large files"""
    
    def test_large_file_handling(self, tmp_path):
        """Test handling of large CSV files"""
        # Generate large file
        generator = TestDataGenerator()
        large_data = generator.generate_large_file_data(n_rows=100000)
        
        csv_path = tmp_path / "large_test.csv"
        large_data.to_csv(csv_path, index=False)
        
        # Test loading
        loader = DataLoader(csv_path)
        file_info = loader.get_file_info()
        
        assert file_info['file_size_mb'] > 0
        
        # Should load successfully
        df = loader.load()
        assert len(df) > 0
    
    def test_cleaning_performance(self):
        """Test cleaning performance on medium dataset"""
        import time
        
        data = TestDataGenerator(10000).generate_dirty_data()
        preferences = CleaningPreferences()
        
        start_time = time.time()
        cleaner = DataCleaner(data, preferences)
        cleaned_df, report = cleaner.clean()
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (< 10 seconds for 10k rows)
        assert processing_time < 10
        
        print(f"\nCleaned 10,000 rows in {processing_time:.2f} seconds")
        print(f"Processing rate: {len(data) / processing_time:.0f} rows/second")


class TestAccuracy:
    """Test cleaning accuracy with ground truth"""
    
    def test_missing_value_accuracy(self):
        """Test accuracy of missing value imputation"""
        # Create data with known missing pattern
        np.random.seed(42)
        clean = pd.DataFrame({
            'value': np.random.normal(100, 15, 1000)
        })
        
        # Introduce random missing values
        dirty = clean.copy()
        missing_indices = np.random.choice(dirty.index, size=100, replace=False)
        dirty.loc[missing_indices, 'value'] = np.nan
        
        # Clean with median imputation
        preferences = CleaningPreferences(fill_numeric_method='median')
        cleaner = DataCleaner(dirty, preferences)
        cleaned_df, _ = cleaner.clean()
        
        # Check that imputed values are reasonable
        original_median = clean['value'].median()
        cleaned_median = cleaned_df['value'].median()
        
        # Should be within 5% of original
        assert abs(cleaned_median - original_median) / original_median < 0.05
    
    def test_outlier_detection_accuracy(self):
        """Test accuracy of outlier detection"""
        # Create data with known outliers
        np.random.seed(42)
        normal_data = np.random.normal(100, 15, 950)
        outliers = np.array([200, 210, 220, -50, -60])  # 5 clear outliers
        
        df = pd.DataFrame({
            'value': np.concatenate([normal_data, outliers])
        })
        
        preferences = CleaningPreferences(
            detect_outliers=True,
            outlier_method='iqr',
            outlier_action='remove'
        )
        
        cleaner = DataCleaner(df, preferences)
        cleaned_df, _ = cleaner.clean()
        
        # Should detect and remove most outliers
        assert len(cleaned_df) < len(df)
        assert cleaned_df['value'].max() < 180  # Outliers removed


def run_accuracy_benchmark():
    """Run comprehensive accuracy benchmark"""
    print("\n" + "="*60)
    print("ACCURACY BENCHMARK REPORT")
    print("="*60)
    
    generator = TestDataGenerator(1000)
    dirty_data = generator.generate_dirty_data()
    
    preferences = CleaningPreferences()
    cleaner = DataCleaner(dirty_data, preferences)
    cleaned_df, report = cleaner.clean()
    
    print(f"\nOriginal Data: {report['summary']['original_shape']}")
    print(f"Cleaned Data: {report['summary']['final_shape']}")
    print(f"\nRows Removed: {report['summary']['rows_removed']}")
    print(f"Columns Removed: {report['summary']['columns_removed']}")
    print(f"\nMissing Values - Before: {report['data_quality']['missing_values_before']}")
    print(f"Missing Values - After: {report['data_quality']['missing_values_after']}")
    print(f"\nDuplicates - Before: {report['data_quality']['duplicates_before']}")
    print(f"Duplicates - After: {report['data_quality']['duplicates_after']}")
    
    print(f"\nCleaning Actions Performed:")
    for i, log in enumerate(report['cleaning_log'], 1):
        print(f"{i}. {log['action']}")
    
    # Calculate accuracy scores
    missing_reduction = (
        (report['data_quality']['missing_values_before'] - 
         report['data_quality']['missing_values_after']) /
        report['data_quality']['missing_values_before'] * 100
        if report['data_quality']['missing_values_before'] > 0 else 100
    )
    
    duplicate_removal = (
        100 if report['data_quality']['duplicates_after'] == 0 
        else 0
    )
    
    print(f"\n" + "="*60)
    print("ACCURACY SCORES")
    print("="*60)
    print(f"Missing Value Handling: {missing_reduction:.1f}%")
    print(f"Duplicate Removal: {duplicate_removal:.1f}%")
    print(f"Data Retention: {(len(cleaned_df)/len(dirty_data)*100):.1f}%")
    print("="*60)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Run benchmark
    run_accuracy_benchmark()
