import polars as pl
from typing import Type, Dict, Any, List
from vectorguard.contract import DataContract

class ValidationError(Exception):
    """Exception raised when a DataFrame fails contract compliance."""
    def __init__(self, errors: Dict[str, Any]):
        self.errors = errors
        super().__init__(f"Contract validation failed: {errors}")


class ValidationEngine:
    @staticmethod
    def validate(df: pl.DataFrame, contract: Type[DataContract]) -> pl.DataFrame:
        """
        Validates a Polars DataFrame against a DataContract using vector expressions.
        Returns the original DataFrame if all checks pass, otherwise raises ValidationError.
        """
        fields = contract.get_fields()
        errors: Dict[str, Any] = {}
        
        # 1. Structural Validation: Check missing columns
        missing_cols = [col for col in fields if col not in df.columns]
        if missing_cols:
            raise ValidationError({"structure": f"Missing required columns: {missing_cols}"})

        # 2. Build explicit Polars validation expressions
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
            return df

        # Execute all calculations in parallel across columns (Lazy mode)
        # This triggers a single, unified execution pass over the memory buffers
        result_df = df.lazy().select(validation_exprs).collect()

        # 3. Analyze execution matrices for failures
        for col_name, field in fields.items():
            col_errors = {}
            
            # Check individual rule results by checking if any row evaluated to True (failed)
            for check_col in result_df.columns:
                if check_col.startswith(f"{col_name}_") and result_df[check_col].any():
                    # Count how many rows failed this specific assertion
                    fail_count = result_df[check_col].sum()
                    col_errors[check_col.split("_")[-2]] = f"Failed {fail_count} rows"
            
            if col_errors:
                errors[col_name] = col_errors

        if errors:
            raise ValidationError(errors)

        return df
