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
    def validate(
        df: pl.DataFrame,
        contract: Type[DataContract],
        isolate: bool = False
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Validates a Polars DataFrame against a DataContract using vector expressions.

        If isolate=False (default): Raises ValidationError on failure.
        If isolate=True: Returns a tuple of (clean_df, isolated_df) without crashing.
        The isolated dataset includes a '_vg_errors' column describing the
        validation rules that failed for each row.
        """
        fields = contract.get_fields()
        errors: Dict[str, Any] = {}

        # 1. Structural Validation:
        # Check missing columns
        missing_cols = [col for col in fields if col not in df.columns]
        if missing_cols:
            raise ValidationError(
                {"structure": f"Missing required columns: {missing_cols}"}
            )

        # 2. Data Cleaning step
        # Standardize formats and resolve data types natively in memory
        # Custom field-level repair rules are also applied by SafeCoercer.
        df = SafeCoercer.coerce(df, contract)

        # 3. Build explicit Polars validation expressions
        validation_exprs: List[pl.Expr] = []

        for col_name, field in fields.items():
            # Basic constraint: Null checks
            if not field.nullable:
                # Creates a boolean mask where True = invalid (it is null)
                validation_exprs.append(
                    pl.col(col_name)
                    .is_null()
                    .alias(f"{col_name}_null_fail")
                )

            # Numeric Boundary Constraints
            if field.gt is not None:
                validation_exprs.append(
                    (pl.col(col_name) <= field.gt)
                    .alias(f"{col_name}_gt_fail")
                )

            if field.ge is not None:
                validation_exprs.append(
                    (pl.col(col_name) < field.ge)
                    .alias(f"{col_name}_ge_fail")
                )

            if field.lt is not None:
                validation_exprs.append(
                    (pl.col(col_name) >= field.lt)
                    .alias(f"{col_name}_lt_fail")
                )

            if field.le is not None:
                validation_exprs.append(
                    (pl.col(col_name) > field.le)
                    .alias(f"{col_name}_le_fail")
                )

            # String constraints
            if field.length is not None:
                validation_exprs.append(
                    (
                        pl.col(col_name).str.len_chars() != field.length
                    ).alias(f"{col_name}_length_fail")
                )

            if field.regex is not None:
                validation_exprs.append(
                    (
                        pl.col(col_name)
                        .str.contains(field.regex)
                        .not_()
                    ).alias(f"{col_name}_regex_fail")
                )

        # If there are no rules specified, skip computation
        if not validation_exprs:
            if isolate:
                empty_isolated = (
                    df
                    .with_columns(pl.lit("").alias("_vg_errors"))
                    .filter(pl.lit(False))
                )
                return df, empty_isolated

            return df

        # Execute evaluation matrix in a single pass over the data
        matrix_df = df.lazy().select(validation_exprs).collect()

        # 4. Determine validity directly from the boolean evaluation matrix(Strict Vector Check)
        # This is 100% robust against string padding behavior
        row_failed_mask = pl.any_horizontal([pl.col(c) for c in matrix_df.columns])
        is_invalid_series = matrix_df.select(row_failed_mask).to_series()

        # Create descriptive row-by-row error tracing vector
        # Each validation flag is converted into a human-readable rule marker.
        error_trace_exprs: List[pl.Expr] = []

        for check_col in matrix_df.columns:
            # If the boolean rule flag is True, yield the clean name of the violation rule
            rule_name = check_col.removesuffix("_fail")

            error_trace_exprs.append(
                pl.when(pl.col(check_col))
                .then(pl.lit(f"[{rule_name}]"))
                .otherwise(pl.lit(""))
            )

        # Horizontally concatenate all triggered error strings for every row
        # Empty rule markers are discarded by using an empty string as the fallback.
        error_strings_df = (
            matrix_df
            .lazy()
            .with_columns(
                pl.concat_str(
                    error_trace_exprs,
                    separator=" "
                ).str.strip_chars() # Remove any hanging spaces from empty rules
                .alias("_vg_errors")
            )
            .select("_vg_errors")
            .collect()
        )

        # Clean the string by checking whether at least one rule produced an error.
        is_invalid_series = (
            error_strings_df["_vg_errors"]
            .str.len_chars()
            .gt(0)
        )

        # 5. Isolation Mode Routing
        if isolate:
            # Inject error reason logs exclusively into the isolated outputs
            df_with_err = df.with_columns(
                error_strings_df["_vg_errors"]
            )

            # Clean rows should preserve the original cleaned schema.
            clean_df = (
                df_with_err
                .filter(~is_invalid_series)
                .drop("_vg_errors")
            )

            # Invalid rows retain the detailed validation trace.
            isolated_df = df_with_err.filter(is_invalid_series)

            return clean_df, isolated_df

        # 6. Default Mode: Collect and raise errors if any row failed
        if is_invalid_series.any():
            errors = {}

            for col_name, field in fields.items():
                col_errors = {}

                for check_col in matrix_df.columns:
                    if (
                        check_col.startswith(f"{col_name}_")
                        and matrix_df[check_col].any()
                    ):
                        fail_count = matrix_df[check_col].sum()

                        # Extract the validation rule name from:
                        # <column>_<rule>_fail
                        rule_name = check_col[
                            len(col_name) + 1:
                        ].removesuffix("_fail")

                        col_errors[rule_name] = (
                            f"Failed {fail_count} rows"
                        )

                if col_errors:
                    errors[col_name] = col_errors

            raise ValidationError(errors)

        return df