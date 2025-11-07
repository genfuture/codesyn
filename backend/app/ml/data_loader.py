"""
Advanced Data Loader - Supports multiple formats with big file handling
"""
import pandas as pd
import dask.dataframe as dd
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import chardet
from sqlalchemy import create_engine, inspect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Loads data from various formats with automatic format detection."""
    
    CHUNK_SIZE = 10000  # Default chunk size for large files
    LARGE_FILE_THRESHOLD = 300 * 1024 * 1024  # 300 MB
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.file_size = self.file_path.stat().st_size if self.file_path.exists() else 0
        self.is_large_file = self.file_size > self.LARGE_FILE_THRESHOLD
        self.encoding = None
        
    def detect_encoding(self) -> str:
        """Detect file encoding for text files"""
        if self.encoding:
            return self.encoding
            
        try:
            with open(self.file_path, 'rb') as file:
                raw_data = file.read(100000)  # Read first 100KB
                result = chardet.detect(raw_data)
                self.encoding = result['encoding'] or 'utf-8'
                logger.info(f"Detected encoding: {self.encoding} (confidence: {result['confidence']})")
                return self.encoding
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}. Using utf-8")
            self.encoding = 'utf-8'
            return self.encoding
    
    def load_csv(self, **kwargs) -> Union[pd.DataFrame, dd.DataFrame]:
        """Load CSV file with automatic encoding detection"""
        encoding = kwargs.get('encoding') or self.detect_encoding()
        
        if self.is_large_file:
            logger.info(f"Loading large CSV file ({self.file_size / 1024 / 1024:.2f} MB) with Dask")
            return dd.read_csv(
                self.file_path,
                encoding=encoding,
                blocksize="64MB",
                **{k: v for k, v in kwargs.items() if k != 'encoding'}
            )
        else:
            logger.info(f"Loading CSV file with pandas")
            return pd.read_csv(self.file_path, encoding=encoding, **kwargs)
    
    def load_tsv(self, **kwargs) -> Union[pd.DataFrame, dd.DataFrame]:
        """Load TSV file"""
        kwargs['sep'] = '\t'
        return self.load_csv(**kwargs)
    
    def load_excel(self, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load Excel file"""
        logger.info(f"Loading Excel file: {self.file_path}")
        
        if self.is_large_file:
            logger.warning("Excel file is large. This may take a while...")
        
        # Get all sheet names
        excel_file = pd.ExcelFile(self.file_path)
        sheets = excel_file.sheet_names
        
        if sheet_name:
            return pd.read_excel(self.file_path, sheet_name=sheet_name, **kwargs)
        elif len(sheets) == 1:
            return pd.read_excel(self.file_path, sheet_name=sheets[0], **kwargs)
        else:
            # Return all sheets as a dictionary
            logger.info(f"Found multiple sheets: {sheets}")
            return {sheet: pd.read_excel(self.file_path, sheet_name=sheet, **kwargs) 
                    for sheet in sheets}
    
    def load_json(self, **kwargs) -> Union[pd.DataFrame, Dict]:
        """Load JSON file with smart format detection"""
        encoding = kwargs.get('encoding') or self.detect_encoding()
        
        if self.is_large_file:
            logger.info("Loading large JSON file line by line")
            # Try to load as JSON lines format
            try:
                return pd.read_json(self.file_path, lines=True, encoding=encoding, **kwargs)
            except:
                # Fall back to regular JSON with chunking
                logger.warning("Not a JSON lines file, loading as regular JSON")
        
        with open(self.file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        # Try to convert to DataFrame
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Check if it's a dict of lists (columnar format)
            if all(isinstance(v, list) for v in data.values()):
                return pd.DataFrame(data)
            else:
                # Single record or nested structure
                return pd.DataFrame([data])
        
        return data
    
    def load_xml(self, record_path: Optional[str] = None) -> pd.DataFrame:
        """Load XML file and convert to DataFrame"""
        logger.info(f"Loading XML file: {self.file_path}")
        
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        
        if record_path:
            # User specified the record element path
            records = root.findall(record_path)
        else:
            # Try to auto-detect record elements
            # Assume records are the first repeating child elements
            children = list(root)
            if children:
                first_tag = children[0].tag
                records = root.findall(first_tag)
            else:
                records = [root]
        
        # Convert XML records to dictionaries
        data = []
        for record in records:
            record_dict = self._xml_to_dict(record)
            data.append(record_dict)
        
        return pd.DataFrame(data)
    
    def _xml_to_dict(self, element) -> Dict:
        """Recursively convert XML element to dictionary"""
        result = {}
        
        # Add attributes
        if element.attrib:
            result.update({f"@{k}": v for k, v in element.attrib.items()})
        
        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:  # No children
                return element.text.strip()
            result['#text'] = element.text.strip()
        
        # Add child elements
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                # Multiple elements with same tag
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result
    
    def load_sql(self, connection_string: str, query: Optional[str] = None, 
                 table_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from SQL database"""
        logger.info(f"Connecting to database...")
        
        engine = create_engine(connection_string)
        
        if query:
            logger.info(f"Executing query: {query[:100]}...")
            if self.is_large_file or 'chunksize' in kwargs:
                # Return iterator for large results
                return pd.read_sql_query(query, engine, **kwargs)
            return pd.read_sql_query(query, engine)
        elif table_name:
            logger.info(f"Loading table: {table_name}")
            if self.is_large_file or 'chunksize' in kwargs:
                return pd.read_sql_table(table_name, engine, **kwargs)
            return pd.read_sql_table(table_name, engine)
        else:
            # List available tables
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            logger.info(f"Available tables: {tables}")
            raise ValueError("Please specify either query or table_name parameter")
    
    def load(self, **kwargs) -> Union[pd.DataFrame, dd.DataFrame, Dict]:
        """Auto-detect format and load data"""
        extension = self.file_path.suffix.lower()
        
        loaders = {
            '.csv': self.load_csv,
            '.tsv': self.load_tsv,
            '.txt': self.load_csv,  # Assume CSV-like
            '.xlsx': self.load_excel,
            '.xls': self.load_excel,
            '.json': self.load_json,
            '.jsonl': lambda **kw: self.load_json(lines=True, **kw),
            '.xml': self.load_xml,
        }
        
        if extension in loaders:
            return loaders[extension](**kwargs)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get metadata about the file"""
        return {
            'file_name': self.file_path.name,
            'file_path': str(self.file_path),
            'file_size_mb': round(self.file_size / 1024 / 1024, 2),
            'is_large_file': self.is_large_file,
            'extension': self.file_path.suffix,
            'encoding': self.encoding
        }
