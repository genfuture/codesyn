#!/usr/bin/env python3
"""
Better Quick Start with Adjusted Settings
Shows more reasonable cleaning results
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from app.ml.data_loader import DataLoader
from app.ml.questionnaire import CleaningQuestionnaire, CleaningPreferences, CleaningStrategy
from app.ml.data_cleaner import DataCleaner

def generate_sample_data():
    """Generate sample dirty dataset"""
    np.random.seed(42)
    
    data = {
        'Name': [
            '  John Doe  ', 'Jane Smith', 'Bob Johnson', None, 'Alice Brown',
            '  Charlie Davis  ', 'Eve Wilson', 'Frank Miller', 'Grace Lee', 'Henry Clark',
            'John Doe', 'Jane Smith'
        ],
        'Age': [
            25, 30, -5, np.nan, 45, 999, 28, 35, 40, 50, 25, 30
        ],
        'Email': [
            'john@example.com', 'jane@test.com', 'invalid-email', 'bob@test.com',
            'alice@example.com', 'charlie@test.com', None, 'frank@test.com',
            'grace@example.com', 'henry@test.com', 'john@example.com', 'jane@test.com'
        ],
        'City': [
            'NEW YORK', 'Los Angeles', 'chicago', 'Houston', 'phoenix',
            'Philadelphia', 'San Antonio', None, 'San Diego', 'Dallas',
            'NEW YORK', 'Los Angeles'
        ],
        'Salary': [
            50000, 60000, 45000, np.nan, 75000, 80000, 55000, 65000, 70000, 85000,
            50000, 60000
        ],
        'Join_Date': [
            '2024-01-15', '01/20/2024', '2024-02-10', '15/03/2024', None,
            '2024-04-05', '05/10/2024', '2024-06-15', None, '2024-08-01',
            '2024-01-15', '01/20/2024'
        ],
        'Active': [
            'yes', 'no', 'true', 'false', '1', '0', 'Y', 'N', None, 'yes',
            'yes', 'no'
        ],
        'Department': [
            'Sales', 'Sales', 'Engineering', 'Sales', 'Engineering',
            'Marketing', 'Sales', 'Engineering', 'Marketing', 'Sales',
            'Sales', 'Sales'
        ]
    }
    
    return pd.DataFrame(data)

def main():
    print("\n" + "="*80)
    print("ADVANCED DATA CLEANING TOOL - BETTER DEMO")
    print("="*80)
    
    # Generate data
    print("\n📊 Generating sample data...")
    df = generate_sample_data()
    df.to_csv('sample_dirty_data.csv', index=False)
    
    print(f"✓ Created: sample_dirty_data.csv")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Missing: {df.isnull().sum().sum()} cells ({df.isnull().sum().sum()/(len(df)*len(df.columns))*100:.1f}%)")
    print(f"  Duplicates: {df.duplicated().sum()} rows")
    
    # Show sample
    print("\n📝 Sample data (first 5 rows):")
    print(df.head().to_string())
    
    # Better preferences - more conservative
    print("\n⚙️  Using CONSERVATIVE cleaning settings...")
    preferences = CleaningPreferences(
        strategy=CleaningStrategy.CONSERVATIVE,
        drop_threshold=90,  # Only drop if >90% missing (not 70%)
        fill_numeric_method='median',
        fill_categorical_method='mode',
        remove_duplicates=True,
        detect_outliers=True,
        outlier_action='flag',  # Just flag, don't remove
        auto_convert_types=True,
        parse_dates=True,
        clean_text=True,
        strip_whitespace=True,
        standardize_column_names=True,
        standardize_case='title'  # Title case for text
    )
    
    # Clean
    print("🧹 Cleaning data...")
    cleaner = DataCleaner(df, preferences)
    cleaned_df, report = cleaner.clean()
    
    # Save
    cleaned_df.to_csv('sample_cleaned_data.csv', index=False)
    
    # Results
    print("\n✨ CLEANING RESULTS:")
    print("="*80)
    print(f"Original shape: {report['summary']['original_shape']}")
    print(f"Final shape:    {report['summary']['final_shape']}")
    print(f"Rows removed:   {report['summary']['rows_removed']}")
    print(f"Columns kept:   {len(cleaned_df.columns)} / {len(df.columns)}")
    
    print(f"\nData Quality:")
    print(f"  Missing values: {report['data_quality']['missing_values_before']} → {report['data_quality']['missing_values_after']}")
    print(f"  Duplicates:     {report['data_quality']['duplicates_before']} → {report['data_quality']['duplicates_after']}")
    
    print(f"\nActions taken:")
    for i, log in enumerate(report['cleaning_log'], 1):
        print(f"  {i}. {log['action']}")
    
    # Show cleaned data
    print("\n✅ Cleaned data (first 5 rows):")
    print(cleaned_df.head().to_string())
    
    # Compare
    print("\n📊 BEFORE vs AFTER:")
    print("="*80)
    print(f"{'Metric':<20} {'Before':>15} {'After':>15} {'Change':>15}")
    print("-"*80)
    print(f"{'Rows':<20} {len(df):>15} {len(cleaned_df):>15} {len(cleaned_df)-len(df):>15}")
    print(f"{'Columns':<20} {len(df.columns):>15} {len(cleaned_df.columns):>15} {len(cleaned_df.columns)-len(df.columns):>15}")
    print(f"{'Missing cells':<20} {df.isnull().sum().sum():>15} {cleaned_df.isnull().sum().sum():>15} {cleaned_df.isnull().sum().sum()-df.isnull().sum().sum():>15}")
    print(f"{'Duplicates':<20} {df.duplicated().sum():>15} {cleaned_df.duplicated().sum():>15} {cleaned_df.duplicated().sum()-df.duplicated().sum():>15}")
    
    orig_mem = df.memory_usage(deep=True).sum() / 1024
    clean_mem = cleaned_df.memory_usage(deep=True).sum() / 1024
    print(f"{'Memory (KB)':<20} {orig_mem:>15.1f} {clean_mem:>15.1f} {clean_mem-orig_mem:>15.1f}")
    
    print("\n" + "="*80)
    print("✅ SUCCESS! Files created:")
    print("  • sample_dirty_data.csv    (original)")
    print("  • sample_cleaned_data.csv  (cleaned)")
    print("="*80)
    
    print("\n💡 TIP: Try different cleaning strategies:")
    print("  Conservative - Keeps most data, minimal changes")
    print("  Moderate     - Balanced approach (recommended)")
    print("  Aggressive   - Maximum cleaning, strict quality")
    
    print("\n🚀 Next steps:")
    print("  1. python backend/main.py --ui                           # Web interface")
    print("  2. python backend/main.py sample_dirty_data.csv -i       # Interactive")
    print("  3. pytest tests/test_cleaning_accuracy.py -v             # Run tests")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
