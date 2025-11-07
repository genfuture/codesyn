#!/usr/bin/env python3
"""
Quick Start Script - Generate sample data and demonstrate cleaning
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from app.ml.data_loader import DataLoader
from app.ml.questionnaire import CleaningQuestionnaire, CleaningPreferences
from app.ml.data_cleaner import DataCleaner

def generate_sample_data():
    """Generate a sample dirty dataset for demo"""
    np.random.seed(42)
    
    data = {
        'Name': [
            '  John Doe  ', 'Jane Smith', 'Bob Johnson', None, 'Alice Brown',
            '  Charlie Davis  ', 'Eve Wilson', 'Frank Miller', 'Grace Lee', 'Henry Clark',
            'John Doe', 'Jane Smith'  # Duplicates
        ],
        'Age': [
            25, 30, -5, np.nan, 45, 999, 28, 35, 40, 50, 25, 30  # Outliers and missing
        ],
        'Email': [
            'john@example.com', 'jane@test.com', 'invalid-email', 'bob@test.com',
            'alice@example.com', 'charlie@test.com', None, 'frank@test.com',
            'grace@example.com', 'henry@test.com', 'john@example.com', 'jane@test.com'
        ],
        'City': [
            'NEW YORK', 'Los Angeles', 'chicago', 'Houston', 'phoenix',
            'Philadelphia', 'San Antonio', None, 'San Diego', 'Dallas',
            'NEW YORK', 'Los Angeles'  # Inconsistent case
        ],
        'Salary': [
            50000, 60000, 45000, np.nan, 75000, 80000, 55000, 65000, 70000, 85000,
            50000, 60000
        ],
        'Join_Date': [
            '2024-01-15', '01/20/2024', '2024-02-10', '15/03/2024', None,
            '2024-04-05', '05/10/2024', '2024-06-15', None, '2024-08-01',
            '2024-01-15', '01/20/2024'  # Different date formats
        ],
        'Active': [
            'yes', 'no', 'true', 'false', '1', '0', 'Y', 'N', None, 'yes',
            'yes', 'no'  # Different boolean representations
        ],
        'Department': [
            'Sales', 'Sales', 'Engineering', 'Sales', 'Engineering',
            'Marketing', 'Sales', 'Engineering', 'Marketing', 'Sales',
            'Sales', 'Sales'
        ]
    }
    
    df = pd.DataFrame(data)
    return df

def main():
    print("=" * 70)
    print("ADVANCED DATA CLEANING TOOL - QUICK START DEMO")
    print("=" * 70)
    
    # Generate sample data
    print("\n1. Generating sample dirty data...")
    df = generate_sample_data()
    
    # Save to CSV
    sample_file = Path('sample_dirty_data.csv')
    df.to_csv(sample_file, index=False)
    print(f"   ✓ Saved sample data to: {sample_file}")
    print(f"   Shape: {df.shape}")
    print(f"   Duplicates: {df.duplicated().sum()}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    # Profile the data
    print("\n2. Profiling data...")
    questionnaire = CleaningQuestionnaire(df)
    summary = questionnaire.get_summary()
    
    print(f"   ✓ Total rows: {summary['total_rows']}")
    print(f"   ✓ Total columns: {summary['total_columns']}")
    print(f"   ✓ Missing percentage: {summary['missing_percentage']:.2f}%")
    print(f"   ✓ Duplicate rows: {summary['duplicate_rows']}")
    
    print("\n   Column Types:")
    for dtype, count in summary['column_types'].items():
        if count > 0:
            print(f"     - {dtype}: {count}")
    
    # Show some issues
    print("\n3. Detected Issues:")
    for col, info in summary['columns'].items():
        if info['suggestions']:
            print(f"\n   {col}:")
            for suggestion in info['suggestions'][:2]:  # Show first 2
                print(f"     • {suggestion}")
    
    # Clean the data
    print("\n4. Cleaning data with default settings...")
    preferences = CleaningPreferences()
    cleaner = DataCleaner(df, preferences)
    cleaned_df, report = cleaner.clean()
    
    print(f"   ✓ Cleaning complete!")
    
    # Show results
    print("\n5. Cleaning Results:")
    print(f"   Original shape: {report['summary']['original_shape']}")
    print(f"   Final shape: {report['summary']['final_shape']}")
    print(f"   Rows removed: {report['summary']['rows_removed']}")
    print(f"   Columns removed: {report['summary']['columns_removed']}")
    
    print("\n   Data Quality Improvements:")
    print(f"   Missing values: {report['data_quality']['missing_values_before']} → "
          f"{report['data_quality']['missing_values_after']}")
    print(f"   Duplicates: {report['data_quality']['duplicates_before']} → "
          f"{report['data_quality']['duplicates_after']}")
    
    print("\n   Actions Performed:")
    for i, log in enumerate(report['cleaning_log'], 1):
        print(f"   {i}. {log['action']}")
    
    # Save cleaned data
    cleaned_file = Path('sample_cleaned_data.csv')
    cleaned_df.to_csv(cleaned_file, index=False)
    print(f"\n6. Saved Results:")
    print(f"   ✓ Cleaned data: {cleaned_file}")
    
    # Show before/after sample
    print("\n7. Before/After Comparison (first 3 rows):")
    print("\n   BEFORE:")
    print(df.head(3).to_string())
    print("\n   AFTER:")
    print(cleaned_df.head(3).to_string())
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Launch Web UI:    python backend/main.py --ui")
    print("2. Interactive CLI:  python backend/main.py sample_dirty_data.csv --interactive")
    print("3. Run Tests:        pytest tests/test_cleaning_accuracy.py -v")
    print("4. View Docs:        cat README.md")
    print("\nFiles created:")
    print(f"  • {sample_file}")
    print(f"  • {cleaned_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
