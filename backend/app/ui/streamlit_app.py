"""
Streamlit UI for Advanced Data Cleaning Tool
Provides interactive interface for viewing and cleaning data
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
from io import BytesIO

# Add backend to path - fix for Streamlit
backend_path = str(Path(__file__).parent.parent.parent)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.ml.data_loader import DataLoader
from app.ml.questionnaire import CleaningQuestionnaire, CleaningPreferences, CleaningStrategy
from app.ml.data_cleaner import DataCleaner

st.set_page_config(
    page_title="Advanced Data Cleaning Tool",
    page_icon="🧹",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'questionnaire' not in st.session_state:
    st.session_state.questionnaire = None
if 'preferences' not in st.session_state:
    st.session_state.preferences = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'answers' not in st.session_state:
    st.session_state.answers = {}


def main():
    st.title("🧹 Advanced Data Cleaning Tool")
    st.markdown("### Support for CSV, SQL, Excel, JSON, XML, and TSV files")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "📁 Load Data",
        "📊 Data Profile",
        "❓ Cleaning Questionnaire",
        "🧹 Clean Data",
        "📈 View Results",
        "💾 Export Data"
    ])
    
    if page == "📁 Load Data":
        load_data_page()
    elif page == "📊 Data Profile":
        data_profile_page()
    elif page == "❓ Cleaning Questionnaire":
        questionnaire_page()
    elif page == "🧹 Clean Data":
        clean_data_page()
    elif page == "📈 View Results":
        results_page()
    elif page == "💾 Export Data":
        export_page()


def load_data_page():
    st.header("📁 Load Data")
    
    # File upload or SQL connection
    data_source = st.radio("Data Source", ["Upload File", "SQL Database"])
    
    if data_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'tsv', 'xlsx', 'xls', 'json', 'jsonl', 'xml']
        )
        
        if uploaded_file:
            # Show upload info
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📤 Uploading: {uploaded_file.name} ({file_size_mb:.2f} MB)")
            
            if file_size_mb > 50:
                st.warning("⏳ Large file detected. This may take 1-2 minutes to upload and process. Please be patient...")
            
            # Save to temp file with progress
            temp_path = Path(f"/tmp/{uploaded_file.name}")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Saving file...")
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                progress_bar.progress(25)
                
                status_text.text("Loading data...")
                with st.spinner("Reading file format..."):
                    loader = DataLoader(temp_path)
                    
                    # Show file info
                    file_info = loader.get_file_info()
                    st.success(f"""
                    ✅ **File:** {file_info['file_name']}  
                    **Size:** {file_info['file_size_mb']} MB  
                    **Type:** {file_info['extension']}  
                    **Large File:** {'Yes' if file_info['is_large_file'] else 'No'}
                    """)
                    progress_bar.progress(50)
                    
                    # Load based on file type
                    status_text.text("Parsing data...")
                    if temp_path.suffix == '.xlsx' or temp_path.suffix == '.xls':
                        # Excel with sheet selection
                        excel_file = pd.ExcelFile(temp_path)
                        sheet = st.selectbox("Select Sheet", excel_file.sheet_names)
                        df = loader.load_excel(sheet_name=sheet)
                    else:
                        df = loader.load()
                    progress_bar.progress(75)
                    
                    # Convert Dask to Pandas if needed (for large files, compute sample)
                    status_text.text("Processing large file...")
                    if hasattr(df, 'compute'):
                        st.warning("⚠️ Large file detected. Loading sample (50,000 rows) for preview and cleaning...")
                        df = df.head(50000).compute()
                    
                    # For very large pandas dataframes, also sample
                    elif len(df) > 100000:
                        st.warning(f"⚠️ Large dataset ({len(df):,} rows). Sampling 50,000 rows for performance...")
                        df = df.sample(n=50000, random_state=42)
                    
                    progress_bar.progress(90)
                    status_text.text("Finalizing...")
                    
                    st.session_state.df = df
                    st.session_state.questionnaire = CleaningQuestionnaire(df)
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
                    st.dataframe(df.head(10))
                    
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error loading file: {str(e)}")
                st.exception(e)
    
    else:  # SQL Database
        st.subheader("SQL Database Connection")
        
        db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "SQLite"])
        
        if db_type == "SQLite":
            db_file = st.text_input("Database File Path")
            connection_string = f"sqlite:///{db_file}"
        else:
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host", "localhost")
                database = st.text_input("Database Name")
            with col2:
                port = st.number_input("Port", value=5432 if db_type == "PostgreSQL" else 3306)
                username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if db_type == "PostgreSQL":
                connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            else:
                connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        
        query_or_table = st.radio("Load by", ["Table Name", "SQL Query"])
        
        if query_or_table == "Table Name":
            table_name = st.text_input("Table Name")
            if st.button("Load Table"):
                try:
                    loader = DataLoader("")  # Dummy path for SQL
                    df = loader.load_sql(connection_string, table_name=table_name)
                    st.session_state.df = df
                    st.session_state.questionnaire = CleaningQuestionnaire(df)
                    st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
                    st.dataframe(df.head(10))
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            query = st.text_area("SQL Query")
            if st.button("Execute Query"):
                try:
                    loader = DataLoader("")
                    df = loader.load_sql(connection_string, query=query)
                    st.session_state.df = df
                    st.session_state.questionnaire = CleaningQuestionnaire(df)
                    st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
                    st.dataframe(df.head(10))
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def data_profile_page():
    st.header("📊 Data Profile")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please load data first!")
        return
    
    if st.session_state.questionnaire is None:
        st.session_state.questionnaire = CleaningQuestionnaire(st.session_state.df)
    
    summary = st.session_state.questionnaire.get_summary()
    
    # Overall stats
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{summary['total_rows']:,}")
    col2.metric("Total Columns", summary['total_columns'])
    col3.metric("Memory Usage", f"{summary['memory_usage_mb']:.2f} MB")
    col4.metric("Missing %", f"{summary['missing_percentage']:.2f}%")
    
    # Missing values visualization
    if summary['total_missing_cells'] > 0:
        st.subheader("Missing Values by Column")
        missing_data = st.session_state.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        fig = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            labels={'x': 'Column', 'y': 'Missing Count'},
            title="Missing Values Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Column types distribution
    st.subheader("Column Types Distribution")
    type_counts = summary['column_types']
    fig = px.pie(
        values=list(type_counts.values()),
        names=list(type_counts.keys()),
        title="Data Types"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed column information
    st.subheader("Column Details")
    
    for col, info in summary['columns'].items():
        with st.expander(f"📋 {col} ({info['type']})"):
            col1, col2 = st.columns(2)
            col1.write(f"**Missing:** {info['missing']}")
            col2.write(f"**Unique Values:** {info['unique']}")
            
            if info['suggestions']:
                st.write("**Suggested Actions:**")
                for suggestion in info['suggestions']:
                    st.write(f"- {suggestion}")
            
            # Show sample values
            sample = st.session_state.df[col].dropna().head(5).tolist()
            st.write(f"**Sample Values:** {sample}")
    
    # Duplicates
    if summary['duplicate_rows'] > 0:
        st.warning(f"⚠️ Found {summary['duplicate_rows']} duplicate rows")


def questionnaire_page():
    st.header("❓ Cleaning Questionnaire")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please load data first!")
        return
    
    if st.session_state.questionnaire is None:
        st.session_state.questionnaire = CleaningQuestionnaire(st.session_state.df)
    
    st.write("Answer these questions to customize the cleaning process:")
    
    questions = st.session_state.questionnaire.generate_questions()
    answers = {}
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}")
        
        if q['type'] == 'choice':
            answer = st.radio(
                q['question'],
                options=[opt[0] for opt in q['options']],
                format_func=lambda x: next(opt[1] for opt in q['options'] if opt[0] == x),
                index=[opt[0] for opt in q['options']].index(q['default']),
                key=f"q_{q['id']}"
            )
            answers[q['id']] = answer
        
        elif q['type'] == 'boolean':
            answer = st.checkbox(q['question'], value=q['default'], key=f"q_{q['id']}")
            answers[q['id']] = answer
        
        elif q['type'] == 'number':
            answer = st.number_input(
                q['question'],
                min_value=q.get('min', 0),
                max_value=q.get('max', 100),
                value=q['default'],
                key=f"q_{q['id']}"
            )
            answers[q['id']] = answer
    
    st.session_state.answers = answers
    
    if st.button("💾 Save Preferences", type="primary"):
        preferences = st.session_state.questionnaire.apply_answers(answers)
        st.session_state.preferences = preferences
        st.success("✅ Preferences saved! Go to 'Clean Data' to start cleaning.")


def clean_data_page():
    st.header("🧹 Clean Data")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please load data first!")
        return
    
    if st.session_state.preferences is None:
        st.warning("⚠️ Please complete the questionnaire first!")
        return
    
    st.write("Ready to clean your data with the following preferences:")
    
    # Show preferences summary
    prefs = st.session_state.preferences
    st.json({
        "Strategy": prefs.strategy.value,
        "Drop columns with >% missing": prefs.drop_threshold,
        "Fill numeric method": prefs.fill_numeric_method,
        "Remove duplicates": prefs.remove_duplicates,
        "Detect outliers": prefs.detect_outliers,
        "Clean text": prefs.clean_text
    })
    
    if st.button("🚀 Start Cleaning", type="primary"):
        with st.spinner("Cleaning data... This may take a while for large datasets."):
            try:
                cleaner = DataCleaner(st.session_state.df, st.session_state.preferences)
                cleaned_df, report = cleaner.clean()
                
                st.session_state.cleaned_df = cleaned_df
                st.session_state.report = report
                
                st.success("✅ Data cleaning completed!")
                st.balloons()
                
                # Show quick summary
                st.subheader("Cleaning Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Rows Removed", 
                    report['summary']['rows_removed'],
                    delta=f"-{report['summary']['rows_removed']}"
                )
                col2.metric(
                    "Columns Removed",
                    report['summary']['columns_removed'],
                    delta=f"-{report['summary']['columns_removed']}"
                )
                col3.metric(
                    "Memory Saved",
                    f"{report['summary']['memory_saved_mb']:.2f} MB",
                    delta=f"-{report['summary']['memory_saved_mb']:.2f}"
                )
                
                st.info("Go to 'View Results' to see detailed report")
                
            except Exception as e:
                st.error(f"Error during cleaning: {str(e)}")
                st.exception(e)


def results_page():
    st.header("📈 View Results")
    
    if st.session_state.cleaned_df is None:
        st.warning("⚠️ Please clean the data first!")
        return
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cleaned Data", 
        "📋 Cleaning Report", 
        "🔍 Before/After Comparison",
        "📈 Statistics"
    ])
    
    with tab1:
        st.subheader("Cleaned Dataset Preview")
        
        # Limit display for large datasets to prevent 502 errors
        max_rows = 1000
        df_to_show = st.session_state.cleaned_df
        
        if len(df_to_show) > max_rows:
            st.warning(f"⚠️ Dataset has {len(df_to_show):,} rows. Showing first {max_rows:,} rows to prevent timeout.")
            df_to_show = df_to_show.head(max_rows)
        
        st.dataframe(df_to_show, use_container_width=True, height=400)
        
        st.write(f"**Total Rows:** {len(st.session_state.cleaned_df):,}")
        st.write(f"**Shape:** {st.session_state.cleaned_df.shape}")
        st.write(f"**Memory:** {st.session_state.cleaned_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    with tab2:
        st.subheader("Detailed Cleaning Report")
        report = st.session_state.report
        
        # Summary
        st.write("### Summary")
        st.json(report['summary'])
        
        # Data quality
        st.write("### Data Quality Improvements")
        st.json(report['data_quality'])
        
        # Cleaning log
        st.write("### Cleaning Actions")
        for log_entry in report['cleaning_log']:
            with st.expander(f"✓ {log_entry['action']}"):
                st.json(log_entry['details'])
    
    with tab3:
        st.subheader("Before/After Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Data**")
            st.write(f"Shape: {st.session_state.df.shape}")
            st.write(f"Missing: {st.session_state.df.isnull().sum().sum()}")
            st.dataframe(st.session_state.df.head())
        
        with col2:
            st.write("**Cleaned Data**")
            st.write(f"Shape: {st.session_state.cleaned_df.shape}")
            st.write(f"Missing: {st.session_state.cleaned_df.isnull().sum().sum()}")
            st.dataframe(st.session_state.cleaned_df.head())
    
    with tab4:
        st.subheader("Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Data Statistics**")
            st.dataframe(st.session_state.df.describe())
        
        with col2:
            st.write("**Cleaned Data Statistics**")
            st.dataframe(st.session_state.cleaned_df.describe())


def export_page():
    st.header("💾 Export Data")
    
    if st.session_state.cleaned_df is None:
        st.warning("⚠️ Please clean the data first!")
        return
    
    st.write("Export your cleaned data in various formats:")
    
    export_format = st.selectbox(
        "Select Format",
        ["CSV", "Excel (XLSX)", "JSON", "Parquet", "TSV"]
    )
    
    filename = st.text_input("Filename", "cleaned_data")
    
    if st.button("📥 Export"):
        try:
            if export_format == "CSV":
                csv = st.session_state.cleaned_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name=f"{filename}.csv",
                    mime="text/csv"
                )
            
            elif export_format == "Excel (XLSX)":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    st.session_state.cleaned_df.to_excel(writer, index=False)
                
                st.download_button(
                    "Download Excel",
                    buffer.getvalue(),
                    file_name=f"{filename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            elif export_format == "JSON":
                json_str = st.session_state.cleaned_df.to_json(orient='records', indent=2)
                st.download_button(
                    "Download JSON",
                    json_str,
                    file_name=f"{filename}.json",
                    mime="application/json"
                )
            
            elif export_format == "TSV":
                tsv = st.session_state.cleaned_df.to_csv(index=False, sep='\t')
                st.download_button(
                    "Download TSV",
                    tsv,
                    file_name=f"{filename}.tsv",
                    mime="text/tab-separated-values"
                )
            
            elif export_format == "Parquet":
                buffer = BytesIO()
                st.session_state.cleaned_df.to_parquet(buffer, index=False)
                st.download_button(
                    "Download Parquet",
                    buffer.getvalue(),
                    file_name=f"{filename}.parquet",
                    mime="application/octet-stream"
                )
            
            st.success("✅ Export ready!")
            
        except Exception as e:
            st.error(f"Export error: {str(e)}")
    
    # Also allow saving report
    st.subheader("Export Cleaning Report")
    if st.session_state.report and st.button("📄 Download Report"):
        report_json = json.dumps(st.session_state.report, indent=2, default=str)
        st.download_button(
            "Download Report (JSON)",
            report_json,
            file_name="cleaning_report.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
