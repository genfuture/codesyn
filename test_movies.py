#!/usr/bin/env python3
"""
Test the data cleaning tool on movies.csv dataset
Compare before and after results with detailed metrics
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from app.ml.data_loader import DataLoader
from app.ml.questionnaire import CleaningQuestionnaire, CleaningPreferences, CleaningStrategy
from app.ml.data_cleaner import DataCleaner

def analyze_data_quality(df, name="Dataset"):
    """Analyze data quality metrics"""
    print(f"\n{'='*80}")
    print(f"{name} Analysis")
    print(f"{'='*80}")
    
    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"  Total Rows:    {len(df):,}")
    print(f"  Total Columns: {len(df.columns)}")
    print(f"  Memory Usage:  {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Missing values
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    print(f"\n🔍 Missing Values:")
    print(f"  Total Missing: {missing_cells:,} cells ({missing_cells/total_cells*100:.2f}%)")
    
    missing_by_col = df.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    if len(missing_cols) > 0:
        print(f"  Columns with missing data:")
        for col, count in missing_cols.head(300).items():
            pct = count / len(df) * 100
            print(f"    • {col}: {count:,} ({pct:.1f}%)")
    
    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\n📋 Duplicates:")
    print(f"  Duplicate Rows: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")
    
    # Data types
    print(f"\n🏷️  Data Types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count} columns")
    
    # Column info
    print(f"\n📝 Columns:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        unique = df[col].nunique()
        missing = df[col].isnull().sum()
        print(f"  {i}. {col:30} | Type: {str(dtype):10} | Unique: {unique:6,} | Missing: {missing:6,}")
    
    # Sample data
    print(f"\n📄 Sample Data (first 300 rows):")
    print(df.head(300).to_string())
    
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_cells': int(missing_cells),
        'missing_pct': float(missing_cells/total_cells*100),
        'duplicates': int(dup_count),
        'memory_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
    }

def main():
    print("\n" + "="*80)
    print("TESTING DATA CLEANING TOOL ON MOVIES.CSV")
    print("="*80)
    
    # Load the movies dataset
    print("\n🎬 Loading movies.csv...")
    file_path = Path('movies.csv')
    
    if not file_path.exists():
        print(f"❌ Error: movies.csv not found!")
        return
    
    loader = DataLoader(file_path)
    df_original = loader.load()
    
    print(f"✅ Loaded {len(df_original):,} movies")
    
    # Analyze original data
    original_metrics = analyze_data_quality(df_original, "ORIGINAL DATA")
    
    # Profile and get recommendations
    print("\n" + "="*80)
    print("PROFILING DATA & GENERATING RECOMMENDATIONS")
    print("="*80)
    
    questionnaire = CleaningQuestionnaire(df_original)
    summary = questionnaire.get_summary()
    
    print(f"\n🔬 Data Profile Summary:")
    print(f"  Column Types Detected:")
    for dtype, count in summary['column_types'].items():
        if count > 0:
            print(f"    • {dtype}: {count}")
    
    print(f"\n💡 Recommended Actions:")
    action_count = 0
    for col, info in summary['columns'].items():
        if info['suggestions']:
            action_count += len(info['suggestions'])
            print(f"\n  {col}:")
            for suggestion in info['suggestions']:
                print(f"    • {suggestion}")
    
    print(f"\n  Total recommendations: {action_count}")
    
    # Clean with different strategies
    strategies = [
        ('Conservative', CleaningStrategy.CONSERVATIVE, 95),
        ('Moderate', CleaningStrategy.MODERATE, 80),
        ('Aggressive', CleaningStrategy.AGGRESSIVE, 60)
    ]
    
    results = {}
    
    for strategy_name, strategy, threshold in strategies:
        print("\n" + "="*80)
        print(f"CLEANING WITH {strategy_name.upper()} STRATEGY")
        print("="*80)
        
        preferences = CleaningPreferences(
            strategy=strategy,
            drop_threshold=threshold,
            fill_numeric_method='median',
            fill_categorical_method='mode',
            remove_duplicates=True,
            detect_outliers=True,
            outlier_action='flag' if strategy == CleaningStrategy.CONSERVATIVE else 'cap',
            auto_convert_types=True,
            parse_dates=True,
            clean_text=True,
            strip_whitespace=True,
            standardize_column_names=True,
            standardize_case=None
        )
        
        print(f"\n⚙️  Settings:")
        print(f"  Strategy: {strategy_name}")
        print(f"  Drop threshold: {threshold}%")
        print(f"  Outlier action: {preferences.outlier_action}")
        
        # Clean
        cleaner = DataCleaner(df_original.copy(), preferences)
        cleaned_df, report = cleaner.clean()
        
        # Analyze cleaned data
        cleaned_metrics = analyze_data_quality(cleaned_df, f"CLEANED DATA ({strategy_name})")
        
        # Store results
        results[strategy_name] = {
            'df': cleaned_df,
            'report': report,
            'metrics': cleaned_metrics
        }
        
        # Show improvements
        print(f"\n✨ IMPROVEMENTS ({strategy_name}):")
        print(f"  Rows: {original_metrics['rows']:,} → {cleaned_metrics['rows']:,} "
              f"({cleaned_metrics['rows']-original_metrics['rows']:+,})")
        print(f"  Columns: {original_metrics['columns']} → {cleaned_metrics['columns']} "
              f"({cleaned_metrics['columns']-original_metrics['columns']:+})")
        print(f"  Missing: {original_metrics['missing_cells']:,} → {cleaned_metrics['missing_cells']:,} "
              f"({cleaned_metrics['missing_cells']-original_metrics['missing_cells']:+,})")
        print(f"  Missing %: {original_metrics['missing_pct']:.2f}% → {cleaned_metrics['missing_pct']:.2f}% "
              f"({cleaned_metrics['missing_pct']-original_metrics['missing_pct']:+.2f}%)")
        print(f"  Duplicates: {original_metrics['duplicates']:,} → {cleaned_metrics['duplicates']:,} "
              f"({cleaned_metrics['duplicates']-original_metrics['duplicates']:+,})")
        print(f"  Memory: {original_metrics['memory_mb']:.2f} MB → {cleaned_metrics['memory_mb']:.2f} MB "
              f"({cleaned_metrics['memory_mb']-original_metrics['memory_mb']:+.2f} MB)")
        
        print(f"\n📋 Actions Performed:")
        for i, log in enumerate(report['cleaning_log'], 1):
            print(f"  {i}. {log['action']}")
        
        # Save cleaned file
        output_file = f'movies_cleaned_{strategy_name.lower()}.csv'
        cleaned_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved: {output_file}")
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON - ALL STRATEGIES")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Original':>15} {'Conservative':>15} {'Moderate':>15} {'Aggressive':>15}")
    print("-"*80)
    
    metrics_to_compare = [
        ('Rows', 'rows'),
        ('Columns', 'columns'),
        ('Missing Cells', 'missing_cells'),
        ('Missing %', 'missing_pct'),
        ('Duplicates', 'duplicates'),
        ('Memory (MB)', 'memory_mb')
    ]
    
    for metric_name, metric_key in metrics_to_compare:
        orig_val = original_metrics[metric_key]
        cons_val = results['Conservative']['metrics'][metric_key]
        mod_val = results['Moderate']['metrics'][metric_key]
        agg_val = results['Aggressive']['metrics'][metric_key]
        
        if metric_key == 'missing_pct' or metric_key == 'memory_mb':
            print(f"{metric_name:<25} {orig_val:>15.2f} {cons_val:>15.2f} {mod_val:>15.2f} {agg_val:>15.2f}")
        else:
            print(f"{metric_name:<25} {orig_val:>15,} {cons_val:>15,} {mod_val:>15,} {agg_val:>15,}")
    
    # Quality score
    print("\n" + "="*80)
    print("CLEANING EFFECTIVENESS SCORES")
    print("="*80)
    
    for strategy_name in ['Conservative', 'Moderate', 'Aggressive']:
        metrics = results[strategy_name]['metrics']
        
        # Calculate scores
        missing_reduction = (1 - metrics['missing_cells'] / max(original_metrics['missing_cells'], 1)) * 100
        duplicate_reduction = (1 - metrics['duplicates'] / max(original_metrics['duplicates'], 1)) * 100
        data_retention = (metrics['rows'] / original_metrics['rows']) * 100
        
        overall_score = (missing_reduction * 0.4 + duplicate_reduction * 0.3 + data_retention * 0.3)
        
        print(f"\n{strategy_name} Strategy:")
        print(f"  Missing Value Reduction:  {missing_reduction:6.1f}%")
        print(f"  Duplicate Removal:        {duplicate_reduction:6.1f}%")
        print(f"  Data Retention:           {data_retention:6.1f}%")
        print(f"  ───────────────────────────────────")
        print(f"  Overall Effectiveness:    {overall_score:6.1f}% ⭐")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Find best strategy
    best_strategy = 'Moderate'  # Default recommendation
    
    if original_metrics['missing_pct'] > 30:
        best_strategy = 'Aggressive'
        reason = "High missing data requires aggressive cleaning"
    elif original_metrics['missing_pct'] < 10 and original_metrics['duplicates'] < len(df_original) * 0.05:
        best_strategy = 'Conservative'
        reason = "Data is relatively clean, use conservative approach"
    else:
        reason = "Balanced approach works best for this dataset"
    
    print(f"\n✅ Best Strategy: {best_strategy}")
    print(f"   Reason: {reason}")
    print(f"\n   Use this file: movies_cleaned_{best_strategy.lower()}.csv")
    
    print("\n" + "="*80)
    print("FILES CREATED:")
    print("="*80)
    print("  • movies_cleaned_conservative.csv")
    print("  • movies_cleaned_moderate.csv")
    print("  • movies_cleaned_aggressive.csv")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
