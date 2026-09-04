import polars as pl
from typing import Type
from vectorguard.contract import DataContract

class SafeCoercer:
    @staticmethod
    def coerce(df: pl.DataFrame, contract: Type[DataContract]) -> pl.DataFrame:
        """
        Cleans and safely casts DataFrame columns based on contract type declarations.
        Handles common human-introduced discrepancies like 'N/A', comma-decimals, 
        and padding whitespaces using vectorized Polars operations.
        """
        fields = contract.get_fields()
        lazy_df = df.lazy()
        coercion_exprs = []

        # Global list of common human-typed representations of missing values
        COMMON_NULLS = ["", "N/A", "n/a", "NULL", "null", "NaN", "nan", "none", "None"]

        for col_name, field in fields.items():
            if col_name not in df.columns:
                continue  # Skip; structural missing column checks will trigger later

            expected_type = field.data_type
            current_dtype = df.schema[col_name]

            # If current type matches expected semantic types, skip coercion
            if expected_type == int and current_dtype in [pl.Int64, pl.Int32]:
                continue
            if expected_type == float and current_dtype in [pl.Float64, pl.Float32]:
                continue
            if expected_type == str and current_dtype == pl.String:
                continue

            # VECTORIZED COERCION: STRINGS TO NUMERIC TYPES
            if expected_type in [int, float] and current_dtype == pl.String:
                # 1. Strip structural whitespaces from user input
                expr = pl.col(col_name).str.strip_chars()
                
                # 2. Map standard human null strings to native Polars null values
                expr = pl.when(expr.is_in(COMMON_NULLS)).then(None).otherwise(expr)
                
                # 3. Handle decimal notations and cast securely
                if expected_type == int:
                    expr = expr.cast(pl.Int64, strict=False)
                else:
                    # Replace European/Latin-American decimal commas with standard dots
                    expr = expr.str.replace(",", ".", literal=True).cast(pl.Float64, strict=False)
                
                coercion_exprs.append(expr.alias(col_name))
                
            # VECTORIZED COERCION: PRIMITIVES TO STRINGS
            elif expected_type == str and current_dtype != pl.String:
                coercion_exprs.append(pl.col(col_name).cast(pl.String).alias(col_name))
                
            # VECTORIZED COERCION: DOWNCASTING FLOATS TO INT
            elif expected_type == int and current_dtype in [pl.Float64, pl.Float32]:
                coercion_exprs.append(pl.col(col_name).floor().cast(pl.Int64).alias(col_name))

        if not coercion_exprs:
            return df

        # Execute coercion computations concurrently over memory buffers
        return lazy_df.with_columns(coercion_exprs).collect()
