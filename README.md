# Advanced Data Cleaning Tool

A comprehensive, production-ready data cleaning tool that supports multiple file formats with AI-powered cleaning suggestions, interactive questionnaires, and big file handling capabilities.

## 🎯 Features

### Supported Formats
- **CSV** - With automatic encoding detection
- **TSV** - Tab-separated values
- **Excel** (.xlsx, .xls) - Multi-sheet support
- **JSON** - Regular and JSON Lines format
- **XML** - Automatic record detection
- **SQL** - PostgreSQL, MySQL, SQLite databases

### Advanced Capabilities
- ✅ **Interactive Questionnaire** - Asks context-aware questions before cleaning
- ✅ **Big File Support** - Handles files of any size using Dask and chunking
- ✅ **AI-Powered Suggestions** - Smart recommendations based on data profiling
- ✅ **Multiple Interfaces** - CLI, Web UI (Streamlit), and programmatic API
- ✅ **Comprehensive Testing** - Full test suite with accuracy benchmarks
- ✅ **Detailed Reporting** - Complete audit trail of all cleaning actions

### Cleaning Operations
- Missing value detection and imputation
- Duplicate row removal
- Outlier detection (IQR, Z-score, Isolation Forest)
- Data type auto-conversion
- Text normalization and cleaning
- Column name standardization
- Email/phone/URL validation
- Custom rule application

## 🚀 Quick Start

### Installation

```bash
# Navigate to project directory
cd /Users/cdmstudent/Downloads/datasyn

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Web UI (Recommended for beginners)

```bash
# Launch Streamlit interface
python backend/main.py --ui

# Or directly
streamlit run backend/app/ui/streamlit_app.py
```

Then open your browser to http://localhost:8501

#### 2. Interactive CLI Mode

```bash
# Interactive mode with questionnaire
python backend/main.py data.csv --interactive
```

This will:
1. Load and profile your data
2. Ask contextual questions about cleaning preferences
3. Clean the data based on your answers
4. Save cleaned data and detailed report

#### 3. Batch Mode

```bash
# Use default cleaning settings
python backend/main.py input.csv -o output.csv

# Use custom config
python backend/main.py input.xlsx -o output.xlsx -c config.json
```

#### 4. Programmatic Usage

```python
from app.ml.data_loader import DataLoader
from app.ml.questionnaire import CleaningQuestionnaire, CleaningPreferences
from app.ml.data_cleaner import DataCleaner

# Load data
loader = DataLoader("data.csv")
df = loader.load()

# Create questionnaire and get recommendations
questionnaire = CleaningQuestionnaire(df)
summary = questionnaire.get_summary()
print(summary)

# Set preferences
preferences = CleaningPreferences(
    strategy=CleaningStrategy.MODERATE,
    drop_threshold=70,
    remove_duplicates=True,
    detect_outliers=True
)

# Clean data
cleaner = DataCleaner(df, preferences)
cleaned_df, report = cleaner.clean()

# Save results
cleaned_df.to_csv("cleaned_data.csv", index=False)
```

## 📊 Data Loading Examples

### CSV/TSV
```python
loader = DataLoader("data.csv")
df = loader.load()  # Auto-detects encoding
```

### Excel
```python
loader = DataLoader("data.xlsx")
df = loader.load_excel(sheet_name="Sheet1")
# Or load all sheets
all_sheets = loader.load_excel()
```

### JSON
```python
loader = DataLoader("data.json")
df = loader.load_json()
# For JSON Lines format
df = loader.load_json(lines=True)
```

### XML
```python
loader = DataLoader("data.xml")
df = loader.load_xml(record_path=".//record")
```

### SQL Database
```python
# PostgreSQL
connection = "postgresql://user:pass@localhost:5432/dbname"
loader = DataLoader("")
df = loader.load_sql(connection, table_name="users")

# Or with query
df = loader.load_sql(connection, query="SELECT * FROM users WHERE active=true")
```

## 🧪 Testing

### Run All Tests
```bash
# Run test suite
pytest tests/test_cleaning_accuracy.py -v

# Run with coverage
pytest tests/test_cleaning_accuracy.py -v --cov=backend/app
```

### Run Accuracy Benchmark
```bash
# Generate detailed accuracy report
python tests/test_cleaning_accuracy.py
```

This will output:
- Cleaning effectiveness metrics
- Missing value handling accuracy
- Duplicate removal success rate
- Data retention percentage

### Test Results Expected:
- ✅ Duplicate removal: 100% accuracy
- ✅ Missing value handling: >90% reduction
- ✅ Outlier detection: IQR/Z-score/Isolation Forest
- ✅ Type conversion: Auto-detect and convert
- ✅ Performance: >1000 rows/second on medium datasets

## 📁 Project Structure

```
datasyn/
├── backend/
│   ├── app/
│   │   ├── ml/
│   │   │   ├── data_loader.py      # Multi-format data loading
│   │   │   ├── questionnaire.py    # Interactive questionnaire
│   │   │   ├── data_cleaner.py     # Core cleaning engine
│   │   │   └── ai_cleaning.py      # Original AI cleaning
│   │   └── ui/
│   │       └── streamlit_app.py    # Web interface
│   └── main.py                     # CLI interface
├── tests/
│   └── test_cleaning_accuracy.py   # Comprehensive test suite
├── requirements.txt                # Python dependencies
├── venv/                          # Virtual environment
└── AI_CLEANING_GUIDE.md          # Original guide
```

## ⚙️ Configuration

Create a `config.json` file to customize cleaning behavior:

```json
{
  "strategy": "moderate",
  "drop_threshold": 70,
  "fill_numeric_method": "median",
  "fill_categorical_method": "mode",
  "remove_duplicates": true,
  "detect_outliers": true,
  "outlier_method": "iqr",
  "outlier_action": "flag",
  "auto_convert_types": true,
  "parse_dates": true,
  "clean_text": true,
  "strip_whitespace": true,
  "standardize_column_names": true,
  "column_name_case": "snake"
}
```

## 🎯 Cleaning Strategies

### Conservative
- Minimal changes
- Only remove obvious duplicates
- Gentle outlier detection
- Preserve maximum data

### Moderate (Recommended)
- Balanced approach
- Remove clear quality issues
- Standard outlier detection
- Good data retention

### Aggressive
- Maximum cleaning
- Remove ambiguous data
- Strict outlier detection
- Prioritize quality over quantity

## 📈 Performance

### Large File Handling
The tool automatically detects and handles large files:
- Files > 100 MB use Dask for chunked processing
- Streaming support for SQL queries
- Memory-efficient processing
- Progress indicators for long operations

### Benchmarks
- Small files (<10 MB): < 1 second
- Medium files (10-100 MB): 1-10 seconds
- Large files (100 MB - 1 GB): 10-60 seconds
- Very large files (>1 GB): Chunked processing

## 🔍 Data Profiling

The tool automatically profiles your data:
- Data type inference (numeric, categorical, datetime, text, boolean)
- Missing value analysis
- Duplicate detection
- Outlier identification
- Column uniqueness
- Memory usage estimation

## 📝 Cleaning Report

Every cleaning operation generates a detailed report:
- Original vs final shape
- Rows/columns removed
- Memory saved
- Data quality metrics
- Complete audit trail of actions
- Before/after statistics

## 🛠️ Advanced Features

### Custom Rules
```python
preferences = CleaningPreferences(
    custom_rules={
        'age_validation': {
            'column': 'age',
            'min': 0,
            'max': 120
        }
    }
)
```

### Email/Phone/URL Validation
```python
preferences = CleaningPreferences(
    validate_emails=True,
    validate_phone_numbers=True,
    validate_urls=True
)
```

### Multiple Outlier Detection Methods
- **IQR**: Inter-Quartile Range (default)
- **Z-Score**: Standard deviation based
- **Isolation Forest**: ML-based anomaly detection

## 🤝 Contributing

To extend the tool:
1. Add new data loaders in `data_loader.py`
2. Extend cleaning operations in `data_cleaner.py`
3. Add new questions in `questionnaire.py`
4. Write tests in `tests/`

## 📄 License

This project is provided as-is for data cleaning purposes.

## 🆘 Troubleshooting

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Memory Issues with Large Files
The tool automatically uses Dask for large files, but you can force it:
```python
import dask.dataframe as dd
df = dd.read_csv("large_file.csv", blocksize="64MB")
```

### Encoding Issues
The tool auto-detects encoding, but you can specify:
```python
loader.load_csv(encoding='latin-1')
```

## 📞 Support

For issues or questions:
1. Check the test suite for examples
2. Review the Streamlit UI for interactive guidance
3. Examine the cleaning report for detailed logs

---

**Built with:** Python 3.11 • Pandas • Dask • Streamlit • Scikit-learn
# codesyn
