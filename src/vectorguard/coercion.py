import polars as pl
from typing import Type
from vectorguard.contract import DataContract

class SafeCoercer:
    # Global list of common human-typed representations of missing values
    COMMON_NULLS = [
        "",
        "N/A",
        "n/a",
        "NULL",
        "null",
        "NaN",
        "nan",
        "none",
        "None",
    ]

    @staticmethod
    def coerce(df: pl.DataFrame, contract: Type[DataContract]) -> pl.DataFrame:
        """
        Cleans and safely casts DataFrame columns based on contract type declarations.
        Applies custom user-defined repair logic before executing standard vector coercions.
        Handles common human-introduced discrepancies like 'N/A', comma-decimals,
        and padding whitespaces using vectorized Polars operations.
        """
        fields = contract.get_fields()
        lazy_df = df.lazy()
        coercion_exprs = []

        for col_name, field in fields.items():
            if col_name not in df.columns:
                continue  # Skip; structural missing column checks will trigger later

            expected_type = field.data_type
            current_dtype = df.schema[col_name]

            # 1. APPLY CUSTOM USER REPAIR RULE IF DEFINED
            # Custom repair logic operates on the original column expression.
            # It should repair values without being responsible for the final type.
            expr = pl.col(col_name)

            if field.repair_with is not None:
                expr = field.repair_with(expr)

            # 2. SKIP COERCION WHEN THE COLUMN ALREADY MATCHES THE EXPECTED TYPE
            # Custom repair expressions are still preserved and applied when defined.
            if field.repair_with is None:
                if expected_type == int and current_dtype in [pl.Int64, pl.Int32]:
                    continue
                if expected_type == float and current_dtype in [pl.Float64, pl.Float32]:
                    continue
                if expected_type == str and current_dtype == pl.String:
                    continue

            # 3. VECTORIZED COERCION: STRINGS TO NUMERIC TYPES
            if expected_type in [int, float] and current_dtype == pl.String:
                # 1. Strip structural whitespaces from user input
                expr = expr.str.strip_chars()

                # 2. Map standard human null strings to native Polars null values
                expr = pl.when(expr.is_in(SafeCoercer.COMMON_NULLS)).then(None).otherwise(expr)

                # 3. Handle decimal notations and cast securely
                if expected_type == int:
                    expr = expr.cast(pl.Int64, strict=False)
                else:
                    # Replace European/Latin-American decimal commas with standard dots
                    expr = (
                        expr
                        .str.replace(",", ".", literal=True)
                        .cast(pl.Float64, strict=False)
                    )

                coercion_exprs.append(expr.alias(col_name))

            # 4. VECTORIZED COERCION: PRIMITIVES TO STRINGS
            elif expected_type == str and current_dtype != pl.String:
                coercion_exprs.append(
                    expr.cast(pl.String).alias(col_name)
                )

            # 5. VECTORIZED COERCION: DOWNCASTING FLOATS TO INT
            elif expected_type == int and current_dtype in [pl.Float64, pl.Float32]:
                coercion_exprs.append(
                    expr.floor().cast(pl.Int64).alias(col_name)
                )

            # 6. APPLY CUSTOM REPAIR RESULT WHEN NO STANDARD COERCION IS REQUIRED
            # This ensures repair_with is not discarded simply because the source
            # column already has the expected physical type.
            elif field.repair_with is not None:
                coercion_exprs.append(expr.alias(col_name))

        if not coercion_exprs:
            return df

        # Execute coercion computations concurrently over memory buffers
        return lazy_df.with_columns(coercion_exprs).collect()
