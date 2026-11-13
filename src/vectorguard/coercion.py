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
    def _normalize_numeric_string(expr: pl.Expr) -> pl.Expr:
        """
        Normalizes numeric strings deterministically without corrupting thousands separators:
        - '1,234.56' -> '1234.56' (commas stripped as thousands separators)
        - '10,5'     -> '10.5'    (single comma converted to decimal point)
        - '1,2,3'    -> unchanged (fails safely to null on strict=False cast)
        """
        has_dot = expr.str.contains(".", literal=True)
        comma_count = expr.str.count_matches(",", literal=True)

        return (
            pl.when(has_dot)
            .then(expr.str.replace_all(",", "", literal=True))
            .when(comma_count == 1)
            .then(expr.str.replace(",", ".", literal=True))
            .otherwise(expr)
        )

    @staticmethod
    def coerce(df: pl.DataFrame, contract: Type[DataContract]) -> pl.DataFrame:
        """
        Cleans and safely casts DataFrame columns based on contract type declarations.
        Applies custom user-defined repair logic before executing standard vector coercions.
        Prevents silent data truncation when converting floating-point values to integers.
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
            expr = pl.col(col_name)
            if field.repair_with is not None:
                expr = field.repair_with(expr)

            # 2. SKIP COERCION WHEN THE COLUMN ALREADY MATCHES THE EXPECTED TYPE
            if field.repair_with is None:
                if expected_type == int and current_dtype == pl.Int64:
                    continue
                if expected_type == float and current_dtype == pl.Float64:
                    continue
                if expected_type == str and current_dtype == pl.String:
                    continue

            # 3. VECTORIZED COERCION: STRINGS TO NUMERIC TYPES
            if expected_type in [int, float] and current_dtype == pl.String:
                expr = expr.str.strip_chars()
                expr = pl.when(expr.is_in(SafeCoercer.COMMON_NULLS)).then(None).otherwise(expr)
                normalized = SafeCoercer._normalize_numeric_string(expr)

                if expected_type == int:
                    # Parse via Float64 first only if exact integer string (e.g. "10.0" -> 10, "10.9" -> null)
                    as_float = normalized.cast(pl.Float64, strict=False)
                    expr = (
                        pl.when(as_float.is_not_null() & (as_float == as_float.floor()) & as_float.is_finite())
                        .then(as_float.cast(pl.Int64, strict=False))
                        .otherwise(None)
                    )
                else:
                    expr = normalized.cast(pl.Float64, strict=False)

                coercion_exprs.append(expr.alias(col_name))

            # 4. VECTORIZED COERCION: INTEGER TO FLOAT (Lossless promotion)
            elif expected_type == float and current_dtype.is_integer():
                coercion_exprs.append(expr.cast(pl.Float64).alias(col_name))

            # 5. VECTORIZED COERCION: FLOAT TO INTEGER (Lossless only; fractional values become null)
            elif expected_type == int and current_dtype.is_float():
                safe_int_expr = (
                    pl.when(expr.is_not_null() & (expr == expr.floor()) & expr.is_finite())
                    .then(expr.cast(pl.Int64, strict=False))
                    .otherwise(None)
                )
                coercion_exprs.append(safe_int_expr.alias(col_name))

            # 6. VECTORIZED COERCION: PRIMITIVES TO STRINGS
            elif expected_type == str and current_dtype != pl.String:
                coercion_exprs.append(expr.cast(pl.String).alias(col_name))

            # 7. APPLY CUSTOM REPAIR RESULT WHEN NO STANDARD COERCION IS REQUIRED
            elif field.repair_with is not None:
                coercion_exprs.append(expr.alias(col_name))

        if not coercion_exprs:
            return df

        return lazy_df.with_columns(coercion_exprs).collect()
