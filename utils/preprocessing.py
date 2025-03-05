import logging
import polars as pl
from typing import Dict, List, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

class SchemaValidationError(Exception):
    """Custom exception for schema validation errors"""
    pass

class DataFrameSchema:
    """Schema definition and validation for a Polars DataFrame"""
    
    def __init__(self, schema: Dict[str, pl.DataType]):
        self.schema = schema
        
    def validate(self, df: pl.DataFrame, force_casting: bool = False) -> Tuple[List[str], Optional[pl.DataFrame]]:
        """
        Validate a DataFrame against the schema and optionally force type casting
        
        Args:
            df: DataFrame to validate
            force_casting: If True, will attempt to cast columns to the schema types with Polars cast(strict=True)
            
        Returns:
            Tuple of (list of validation errors, converted DataFrame if force_casting is True)
        """
        errors = []
        converted_df = None
        
        if force_casting:
            # Create a new DataFrame with forced type conversions
            converted_df = df.clone()
            cast_expressions = []
            
            for col_name, expected_dtype in self.schema.items():
                if col_name in df.columns:
                    actual_dtype = df[col_name].dtype
                    
                    # Only cast if types don't match
                    if str(actual_dtype) != str(expected_dtype):
                        try:
                            cast_expressions.append(
                                pl.col(col_name).cast(expected_dtype, strict=True)
                            )
                            warnings.warn(
                                f"Column '{col_name}' type converted from {actual_dtype} to {expected_dtype}. ",
                                UserWarning
                            )
                        except Exception as e:
                            errors.append(
                                f"Failed to convert column '{col_name}' from {actual_dtype} to {expected_dtype}: {str(e)}"
                            )
            
            if cast_expressions:
                try:
                    converted_df = converted_df.with_columns(cast_expressions)
                except Exception as e:
                    errors.append(f"Error during type conversion: {str(e)}")
        
        # Check for required columns
        missing_columns = set(self.schema.keys()) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
            
        # Validate each column's data type
        df_to_check = converted_df if force_casting else df
        for col_name, expected_dtype in self.schema.items():
            if col_name not in df_to_check.columns:
                continue
                
            col = df_to_check[col_name]
            actual_dtype = col.dtype
            
            # Handle type checking by string comparison instead of using isinstance
            if str(actual_dtype) != str(expected_dtype):
                errors.append(
                    f"Column '{col_name}' has incorrect type. "
                    f"Expected {expected_dtype}, got {actual_dtype}"
                )
        
        return errors, converted_df

def validate_dataframe(df: pl.DataFrame, schema: Dict[str, pl.DataType], 
                      force_casting: bool = False) -> Optional[pl.DataFrame]:
    """
    Validate a DataFrame against a schema and raise an exception if invalid
    
    Args:
        df: DataFrame to validate
        schema: Dictionary mapping column names to DataType enums
        force_casting: If True, will attempt to cast columns to schema types
        
    Returns:
        Converted DataFrame if force_casting is True and conversions were made
        
    Raises:
        SchemaValidationError: If validation fails
    
    Example:
    ```python
    schema = {
        "id": DataType.INTEGER,
        "name": DataType.STRING,
        "score": DataType.FLOAT,
        "created_at": DataType.DATETIME,
        "category": DataType.CATEGORICAL
    }
    
    # Just validate
    validate_dataframe(df, schema)
    
    # Validate and force type casting
    converted_df = validate_dataframe(df, schema, force_casting=True)
    ```
    """
    validator = DataFrameSchema(schema)
    errors, converted_df = validator.validate(df, force_casting)
    
    if errors:
        raise SchemaValidationError("\n".join(errors))
        
    return converted_df if force_casting else None