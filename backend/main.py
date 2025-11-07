"""
Command-Line Interface for Advanced Data Cleaning Tool
"""
import argparse
import sys
from pathlib import Path
import json
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from app.ml.data_loader import DataLoader
from app.ml.questionnaire import CleaningQuestionnaire, CleaningPreferences, CleaningStrategy
from app.ml.data_cleaner import DataCleaner
from colorama import Fore, Style, init

init(autoreset=True)


def print_header():
    """Print application header"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}🧹  Advanced Data Cleaning Tool")
    print(f"{Fore.CYAN}   Support: CSV, Excel, JSON, XML, TSV, SQL")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")


def interactive_mode(file_path: str):
    """Run in interactive mode with questionnaire"""
    print_header()
    
    # Load data
    print(f"{Fore.YELLOW}Loading data from: {file_path}{Style.RESET_ALL}")
    loader = DataLoader(Path(file_path))
    
    file_info = loader.get_file_info()
    print(f"\n{Fore.GREEN}File Info:{Style.RESET_ALL}")
    print(f"  Size: {file_info['file_size_mb']} MB")
    print(f"  Type: {file_info['extension']}")
    print(f"  Large File: {file_info['is_large_file']}")
    
    df = loader.load()
    
    # Handle Dask DataFrames
    if hasattr(df, 'compute'):
        print(f"\n{Fore.YELLOW}Large file detected. Loading sample...{Style.RESET_ALL}")
        df = df.head(100000).compute()
    
    print(f"\n{Fore.GREEN}Loaded: {len(df):,} rows × {len(df.columns)} columns{Style.RESET_ALL}")
    
    # Profile data
    print(f"\n{Fore.YELLOW}Profiling data...{Style.RESET_ALL}")
    questionnaire = CleaningQuestionnaire(df)
    summary = questionnaire.get_summary()
    
    print(f"\n{Fore.GREEN}Data Summary:{Style.RESET_ALL}")
    print(f"  Missing: {summary['missing_percentage']:.2f}%")
    print(f"  Duplicates: {summary['duplicate_rows']}")
    print(f"  Memory: {summary['memory_usage_mb']:.2f} MB")
    
    # Show column types
    print(f"\n{Fore.GREEN}Column Types:{Style.RESET_ALL}")
    for dtype, count in summary['column_types'].items():
        if count > 0:
            print(f"  {dtype}: {count}")
    
    # Ask questions
    print(f"\n{Fore.CYAN}{'='*60}")
    print("CLEANING QUESTIONNAIRE")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    questions = questionnaire.generate_questions()
    answers = {}
    
    for i, q in enumerate(questions, 1):
        print(f"\n{Fore.YELLOW}Question {i}/{len(questions)}:{Style.RESET_ALL}")
        print(f"{q['question']}")
        
        if q['type'] == 'choice':
            for j, (key, label) in enumerate(q['options'], 1):
                print(f"  {j}. {label}")
            
            while True:
                try:
                    choice = input(f"\nEnter choice (1-{len(q['options'])}) [default: {q['default']}]: ").strip()
                    if not choice:
                        answers[q['id']] = q['default']
                        break
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(q['options']):
                        answers[q['id']] = q['options'][choice_idx][0]
                        break
                    else:
                        print(f"{Fore.RED}Invalid choice. Try again.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Invalid input. Try again.{Style.RESET_ALL}")
        
        elif q['type'] == 'boolean':
            while True:
                choice = input(f"Enter yes/no [default: {'yes' if q['default'] else 'no'}]: ").strip().lower()
                if not choice:
                    answers[q['id']] = q['default']
                    break
                elif choice in ['y', 'yes', 'true', '1']:
                    answers[q['id']] = True
                    break
                elif choice in ['n', 'no', 'false', '0']:
                    answers[q['id']] = False
                    break
                else:
                    print(f"{Fore.RED}Invalid input. Try again.{Style.RESET_ALL}")
        
        elif q['type'] == 'number':
            while True:
                try:
                    choice = input(f"Enter number ({q.get('min', 0)}-{q.get('max', 100)}) [default: {q['default']}]: ").strip()
                    if not choice:
                        answers[q['id']] = q['default']
                        break
                    value = float(choice)
                    if q.get('min', 0) <= value <= q.get('max', 100):
                        answers[q['id']] = value
                        break
                    else:
                        print(f"{Fore.RED}Out of range. Try again.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Invalid number. Try again.{Style.RESET_ALL}")
    
    # Apply preferences
    preferences = questionnaire.apply_answers(answers)
    
    # Clean data
    print(f"\n{Fore.CYAN}{'='*60}")
    print("CLEANING DATA")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    print(f"{Fore.YELLOW}Processing...{Style.RESET_ALL}")
    cleaner = DataCleaner(df, preferences)
    cleaned_df, report = cleaner.clean()
    
    # Show results
    print(f"\n{Fore.GREEN}✓ Cleaning Complete!{Style.RESET_ALL}\n")
    
    print(f"{Fore.GREEN}Summary:{Style.RESET_ALL}")
    print(f"  Original: {report['summary']['original_shape']}")
    print(f"  Final: {report['summary']['final_shape']}")
    print(f"  Rows removed: {report['summary']['rows_removed']}")
    print(f"  Columns removed: {report['summary']['columns_removed']}")
    print(f"  Memory saved: {report['summary']['memory_saved_mb']:.2f} MB")
    
    print(f"\n{Fore.GREEN}Data Quality:{Style.RESET_ALL}")
    print(f"  Missing before: {report['data_quality']['missing_values_before']}")
    print(f"  Missing after: {report['data_quality']['missing_values_after']}")
    print(f"  Duplicates before: {report['data_quality']['duplicates_before']}")
    print(f"  Duplicates after: {report['data_quality']['duplicates_after']}")
    
    print(f"\n{Fore.GREEN}Actions Performed:{Style.RESET_ALL}")
    for log in report['cleaning_log']:
        print(f"  ✓ {log['action']}")
    
    # Save results
    output_path = input(f"\n{Fore.YELLOW}Save cleaned data to (e.g., cleaned_data.csv): {Style.RESET_ALL}").strip()
    
    if output_path:
        output_path = Path(output_path)
        ext = output_path.suffix.lower()
        
        if ext == '.csv':
            cleaned_df.to_csv(output_path, index=False)
        elif ext in ['.xlsx', '.xls']:
            cleaned_df.to_excel(output_path, index=False)
        elif ext == '.json':
            cleaned_df.to_json(output_path, orient='records', indent=2)
        elif ext == '.tsv':
            cleaned_df.to_csv(output_path, index=False, sep='\t')
        elif ext == '.parquet':
            cleaned_df.to_parquet(output_path, index=False)
        else:
            cleaned_df.to_csv(output_path, index=False)
        
        print(f"{Fore.GREEN}✓ Saved to: {output_path}{Style.RESET_ALL}")
        
        # Save report
        report_path = output_path.with_suffix('.report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"{Fore.GREEN}✓ Report saved to: {report_path}{Style.RESET_ALL}")


def batch_mode(file_path: str, output_path: str, config_path: Optional[str] = None):
    """Run in batch mode with config file"""
    print_header()
    
    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
        preferences = CleaningPreferences(**config)
    else:
        preferences = CleaningPreferences()  # Use defaults
    
    # Load data
    print(f"{Fore.YELLOW}Loading: {file_path}{Style.RESET_ALL}")
    loader = DataLoader(Path(file_path))
    df = loader.load()
    
    if hasattr(df, 'compute'):
        df = df.compute()
    
    print(f"{Fore.GREEN}Loaded: {len(df):,} rows × {len(df.columns)} columns{Style.RESET_ALL}")
    
    # Clean
    print(f"{Fore.YELLOW}Cleaning...{Style.RESET_ALL}")
    cleaner = DataCleaner(df, preferences)
    cleaned_df, report = cleaner.clean()
    
    # Save
    output_path = Path(output_path)
    ext = output_path.suffix.lower()
    
    if ext == '.csv':
        cleaned_df.to_csv(output_path, index=False)
    elif ext in ['.xlsx', '.xls']:
        cleaned_df.to_excel(output_path, index=False)
    elif ext == '.json':
        cleaned_df.to_json(output_path, orient='records', indent=2)
    elif ext == '.tsv':
        cleaned_df.to_csv(output_path, index=False, sep='\t')
    elif ext == '.parquet':
        cleaned_df.to_parquet(output_path, index=False)
    
    print(f"{Fore.GREEN}✓ Saved to: {output_path}{Style.RESET_ALL}")
    
    # Save report
    report_path = output_path.with_suffix('.report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"{Fore.GREEN}✓ Report: {report_path}{Style.RESET_ALL}")
    print(f"\n{Fore.GREEN}Removed {report['summary']['rows_removed']} rows, "
          f"{report['summary']['columns_removed']} columns{Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(
        description='Advanced Data Cleaning Tool - Clean CSV, Excel, JSON, XML, TSV, SQL data'
    )
    
    parser.add_argument('input', nargs='?', help='Input file path')
    parser.add_argument('-o', '--output', help='Output file path (for batch mode)')
    parser.add_argument('-c', '--config', help='Config file path (JSON)')
    parser.add_argument('-i', '--interactive', action='store_true', 
                       help='Run in interactive mode with questionnaire')
    parser.add_argument('--ui', action='store_true', 
                       help='Launch Streamlit UI')
    
    args = parser.parse_args()
    
    if args.ui:
        print(f"{Fore.YELLOW}Launching Streamlit UI...{Style.RESET_ALL}")
        import subprocess
        import os
        
        # Set PYTHONPATH to include backend directory
        env = os.environ.copy()
        backend_dir = str(Path(__file__).parent)
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{backend_dir}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = backend_dir
        
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            str(Path(__file__).parent / 'app' / 'ui' / 'streamlit_app.py')
        ], env=env)
    elif not args.input:
        parser.error("Input file is required when not using --ui mode")
    elif args.interactive or not args.output:
        interactive_mode(args.input)
    else:
        batch_mode(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
