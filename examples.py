"""
Example usage scripts for the Advanced Data Cleaning Tool
"""

# Example 1: Basic CSV Cleaning
def example_csv_cleaning():
    from app.ml.data_loader import DataLoader
    from app.ml.questionnaire import CleaningQuestionnaire, CleaningPreferences
    from app.ml.data_cleaner import DataCleaner
    
    # Load CSV
    loader = DataLoader("sample_data.csv")
    df = loader.load()
    
    # Quick clean with defaults
    preferences = CleaningPreferences()
    cleaner = DataCleaner(df, preferences)
    cleaned_df, report = cleaner.clean()
    
    # Save
    cleaned_df.to_csv("cleaned_data.csv", index=False)
    print(f"Cleaned {len(df)} rows -> {len(cleaned_df)} rows")


# Example 2: Excel with Custom Preferences
def example_excel_cleaning():
    from app.ml.data_loader import DataLoader
    from app.ml.questionnaire import CleaningPreferences, CleaningStrategy
    from app.ml.data_cleaner import DataCleaner
    
    # Load Excel
    loader = DataLoader("sales_data.xlsx")
    df = loader.load_excel(sheet_name="Q1_2024")
    
    # Custom aggressive cleaning
    preferences = CleaningPreferences(
        strategy=CleaningStrategy.AGGRESSIVE,
        drop_threshold=50,  # Drop columns with >50% missing
        remove_duplicates=True,
        detect_outliers=True,
        outlier_action="remove",
        clean_text=True,
        standardize_case="lower"
    )
    
    cleaner = DataCleaner(df, preferences)
    cleaned_df, report = cleaner.clean()
    
    # Export to multiple formats
    cleaned_df.to_excel("cleaned_sales.xlsx", index=False)
    cleaned_df.to_csv("cleaned_sales.csv", index=False)
    cleaned_df.to_json("cleaned_sales.json", orient="records", indent=2)


# Example 3: SQL Database Cleaning
def example_sql_cleaning():
    from app.ml.data_loader import DataLoader
    from app.ml.data_cleaner import DataCleaner
    from app.ml.questionnaire import CleaningPreferences
    
    # Load from PostgreSQL
    connection = "postgresql://user:password@localhost:5432/mydb"
    loader = DataLoader("")
    df = loader.load_sql(
        connection,
        query="SELECT * FROM customers WHERE created_at >= '2024-01-01'"
    )
    
    # Clean
    preferences = CleaningPreferences(
        validate_emails=True,
        validate_phone_numbers=True
    )
    
    cleaner = DataCleaner(df, preferences)
    cleaned_df, report = cleaner.clean()
    
    # Write back to database or export
    cleaned_df.to_csv("cleaned_customers.csv", index=False)


# Example 4: JSON Data Cleaning
def example_json_cleaning():
    from app.ml.data_loader import DataLoader
    from app.ml.data_cleaner import DataCleaner
    from app.ml.questionnaire import CleaningPreferences
    
    # Load JSON
    loader = DataLoader("api_response.json")
    df = loader.load_json()
    
    # Clean
    preferences = CleaningPreferences(
        auto_convert_types=True,
        parse_dates=True,
        standardize_column_names=True
    )
    
    cleaner = DataCleaner(df, preferences)
    cleaned_df, report = cleaner.clean()
    
    # Save back to JSON
    cleaned_df.to_json("cleaned_data.json", orient="records", indent=2)


# Example 5: Large File Processing
def example_large_file():
    from app.ml.data_loader import DataLoader
    from app.ml.data_cleaner import DataCleaner
    from app.ml.questionnaire import CleaningPreferences
    
    # Load large CSV (will use Dask if >100MB)
    loader = DataLoader("large_dataset.csv")
    df = loader.load()
    
    # For very large files, work with samples
    if hasattr(df, 'compute'):
        print("Large file detected, processing in chunks...")
        df_sample = df.head(100000).compute()
    else:
        df_sample = df
    
    # Clean sample
    preferences = CleaningPreferences()
    cleaner = DataCleaner(df_sample, preferences)
    cleaned_df, report = cleaner.clean()
    
    # Save as Parquet for efficient storage
    cleaned_df.to_parquet("cleaned_data.parquet", index=False)


# Example 6: Interactive Questionnaire
def example_interactive():
    from app.ml.data_loader import DataLoader
    from app.ml.questionnaire import CleaningQuestionnaire
    from app.ml.data_cleaner import DataCleaner
    
    # Load data
    loader = DataLoader("survey_data.csv")
    df = loader.load()
    
    # Create questionnaire
    questionnaire = CleaningQuestionnaire(df)
    
    # Get summary
    summary = questionnaire.get_summary()
    print(f"Dataset: {summary['total_rows']} rows, {summary['total_columns']} columns")
    print(f"Missing: {summary['missing_percentage']:.2f}%")
    
    # Generate questions
    questions = questionnaire.generate_questions()
    
    # In a real app, you'd present these to the user
    # For now, use defaults
    answers = {q['id']: q['default'] for q in questions}
    
    # Apply answers
    preferences = questionnaire.apply_answers(answers)
    
    # Clean
    cleaner = DataCleaner(df, preferences)
    cleaned_df, report = cleaner.clean()
    
    print("\nCleaning Report:")
    for log in report['cleaning_log']:
        print(f"  - {log['action']}")


# Example 7: XML Processing
def example_xml_cleaning():
    from app.ml.data_loader import DataLoader
    from app.ml.data_cleaner import DataCleaner
    from app.ml.questionnaire import CleaningPreferences
    
    # Load XML
    loader = DataLoader("data.xml")
    df = loader.load_xml(record_path=".//item")
    
    # Clean
    preferences = CleaningPreferences()
    cleaner = DataCleaner(df, preferences)
    cleaned_df, report = cleaner.clean()
    
    # Export to CSV for easier analysis
    cleaned_df.to_csv("cleaned_xml_data.csv", index=False)


# Example 8: Batch Processing Multiple Files
def example_batch_processing():
    from pathlib import Path
    from app.ml.data_loader import DataLoader
    from app.ml.data_cleaner import DataCleaner
    from app.ml.questionnaire import CleaningPreferences
    import json
    
    data_dir = Path("data")
    output_dir = Path("cleaned_data")
    output_dir.mkdir(exist_ok=True)
    
    # Process all CSV files
    preferences = CleaningPreferences()
    
    for file_path in data_dir.glob("*.csv"):
        print(f"Processing {file_path.name}...")
        
        loader = DataLoader(file_path)
        df = loader.load()
        
        cleaner = DataCleaner(df, preferences)
        cleaned_df, report = cleaner.clean()
        
        # Save cleaned data
        output_path = output_dir / file_path.name
        cleaned_df.to_csv(output_path, index=False)
        
        # Save report
        report_path = output_dir / f"{file_path.stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"  Saved to {output_path}")


if __name__ == "__main__":
    print("Advanced Data Cleaning Tool - Examples")
    print("Run individual example functions to see usage patterns")
