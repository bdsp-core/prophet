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
        
    def validate(self, df: pl.LazyFrame | pl.DataFrame, force_casting: bool = False) -> Tuple[List[str], Optional[pl.LazyFrame | pl.DataFrame]]:
        """
        Validate a DataFrame or LazyFrame against the schema and optionally force type casting
        
        Args:
            df: DataFrame or LazyFrame to validate
            force_casting: If True, will attempt to cast columns to the schema types with Polars cast(strict=True)
            
        Returns:
            Tuple of (list of validation errors, converted DataFrame/LazyFrame if force_casting is True)
        """
        errors = []
        converted_df = None
        
        df_schema = df.collect_schema()
        df_columns = df_schema.names()
        
        # Check for required columns
        missing_columns = set(self.schema.keys()) - set(df_columns)
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        if force_casting:
            # Create expressions for forced type conversions
            cast_expressions = []
            
            for col_name, expected_dtype in self.schema.items():
                if col_name in df_columns:
                    actual_dtype = df_schema[col_name]
                    
                    # Only cast if types don't match
                    if str(actual_dtype) != str(expected_dtype):
                        try:
                            cast_expressions.append(
                                pl.col(col_name).cast(expected_dtype, strict=True).alias(col_name)
                            )
                            warnings.warn(
                                f"Column '{col_name}' type converted from {actual_dtype} to {expected_dtype}.",
                                UserWarning
                            )
                        except Exception as e:
                            errors.append(
                                f"Failed to convert column '{col_name}' from {actual_dtype} to {expected_dtype}: {str(e)}"
                            )
            
            if cast_expressions:
                try:
                    # Apply type conversions if needed
                    converted_df = df.clone().with_columns(cast_expressions)
                except Exception as e:
                    errors.append(f"Error during type conversion: {str(e)}")
            else:
                # No conversions needed
                converted_df = df.clone()
        
        # Validate each column's data type (skip columns that already failed during force_casting)
        schema_to_check = (converted_df if converted_df is not None else df).collect_schema() if force_casting else df_schema

        for col_name, expected_dtype in self.schema.items():
            if col_name not in df_columns:
                continue

            actual_dtype = schema_to_check[col_name]

            # Handle type checking by string comparison
            if str(actual_dtype) != str(expected_dtype):
                error_msg = (
                    f"Column '{col_name}' has incorrect type. "
                    f"Expected {expected_dtype}, got {actual_dtype}"
                )
                if not any(f"column '{col_name}'" in e.lower() for e in errors):
                    errors.append(error_msg)

        # Reject nulls in join-key columns that are declared in the schema.
        # Only checking schema columns (not every column in the DataFrame) so that
        # extra columns the model doesn't use don't trigger spurious errors.
        active_df = converted_df if converted_df is not None else df
        for col_name in ('id', 'date'):
            if col_name in self.schema and col_name in df_columns:
                null_count = active_df.select(pl.col(col_name).is_null().sum()).item()
                if null_count > 0:
                    errors.append(
                        f"Column '{col_name}' contains {null_count} null value(s). "
                        f"Null keys cause silent row loss in joins — drop or fill them before calling the model."
                    )

        return errors, converted_df

def validate_dataframe(df: pl.DataFrame, schema: Dict[str, pl.DataType], 
                      force_casting: bool = False, name: str = None) -> Optional[pl.DataFrame]:
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
        raise SchemaValidationError(
            (f"'{name}' dataframe had error(s):\n" if name else "") +
            "\n".join(errors)
        )
        
    return converted_df if force_casting else None