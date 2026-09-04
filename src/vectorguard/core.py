import polars as pl
from typing import Type, Dict, Any, List, Tuple, Union
from vectorguard.contract import DataContract
from vectorguard.coercion import SafeCoercer

class ValidationError(Exception):
    """Exception raised when a DataFrame fails contract compliance."""
    def __init__(self, errors: Dict[str, Any]):
        self.errors = errors
        super().__init__(f"Contract validation failed: {errors}")


class ValidationEngine:
    @staticmethod
    def validate(df: pl.DataFrame, 
                 contract: Type[DataContract], 
                 isolate: bool = False) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Validates a Polars DataFrame against a DataContract using vector expressions.

        If isolate=False (default): Raises ValidationError on failure.
        If isolate=True: Returns a tuple of (clean_df, isolated_df) without crashing.
        """
        fields = contract.get_fields()
        errors: Dict[str, Any] = {}
        
        # 1. Structural Validation: 
        # Check missing columns
        missing_cols = [col for col in fields if col not in df.columns]
        if missing_cols:
            raise ValidationError({"structure": f"Missing required columns: {missing_cols}"})

        # 2. Data Cleaning step
        # Standardize formats and resolve data types natively in memory
        df = SafeCoercer.coerce(df, contract)

        # 3. Build explicit Polars validation expressions
        validation_exprs: List[pl.Expr] = []
        
        for col_name, field in fields.items():
            # Basic constraint: Null checks
            if not field.nullable:
                # Creates a boolean mask where True = invalid (it is null)
                validation_exprs.append(
                    pl.col(col_name).is_null().alias(f"{col_name}_null_fail")
                )
            
            # Numeric Boundary Constraints
            if field.gt is not None:
                validation_exprs.append((pl.col(col_name) <= field.gt).alias(f"{col_name}_gt_fail"))
            if field.ge is not None:
                validation_exprs.append((pl.col(col_name) < field.ge).alias(f"{col_name}_ge_fail"))
            if field.lt is not None:
                validation_exprs.append((pl.col(col_name) >= field.lt).alias(f"{col_name}_lt_fail"))
            if field.le is not None:
                validation_exprs.append((pl.col(col_name) > field.le).alias(f"{col_name}_le_fail"))

            # String constraints
            if field.length is not None:
                validation_exprs.append(
                    (pl.col(col_name).str.len_chars() != field.length).alias(f"{col_name}_length_fail")
                )
            if field.regex is not None:
                validation_exprs.append(
                    (pl.col(col_name).str.contains(field.regex).not_()).alias(f"{col_name}_regex_fail")
                )

        # If there are no rules specified, skip computation
        if not validation_exprs:
            return (df, df.filter(pl.lit(False))) if isolate else df

        # Execute evaluation matrix in a single pass over the data
        matrix_df = df.lazy().select(validation_exprs).collect()

        # 4. Generate a combined boolean mask across all failed checks
        # If ANY check fails in a row, that entire row is marked as invalid
        row_failed_mask = pl.any_horizontal([pl.col(c) for c in matrix_df.columns])

        # Inject the failure mask back as a temporary boolean series
        is_invalid_series = matrix_df.select(row_failed_mask).to_series()

        # 5. Isolation Mode Routing
        if isolate:
            # Separate data seamlessly using memory-efficient masking
            clean_df = df.filter(~is_invalid_series)
            isolated_df = df.filter(is_invalid_series)
            return clean_df, isolated_df

        # 6. Default Mode: Collect and raise errors if any row failed
        if is_invalid_series.any():
            errors: Dict[str, Any] = {}
            for col_name, field in fields.items():
                col_errors = {}
                for check_col in matrix_df.columns:
                    if check_col.startswith(f"{col_name}_") and matrix_df[check_col].any():
                        fail_count = matrix_df[check_col].sum()
                        col_errors[check_col.split("_")[-2]] = f"Failed {fail_count} rows"
                if col_errors:
                    errors[col_name] = col_errors
            
            raise ValidationError(errors)

        return df