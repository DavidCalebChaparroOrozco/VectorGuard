import polars as pl

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
        Normalizes numeric strings in a single conditional pass:
        - '1,234.56' -> '1234.56' (commas stripped as thousands separators)
        - '10,5'     -> '10.5'    (comma converted to decimal point)
        - '1,2,3'    -> '1.2.3'   (fails safely to null on strict=False Float64 cast)
        """
        return (
            pl.when(expr.str.contains(".", literal=True))
            .then(expr.str.replace_all(",", "", literal=True))
            .otherwise(expr.str.replace_all(",", ".", literal=True))
        )

    @staticmethod
    def coerce(df: pl.DataFrame, contract: type[DataContract]) -> pl.DataFrame:
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
            expr = pl.col(col_name)

            already_target_numeric = (expected_type is int and current_dtype.is_integer()) or (
                expected_type is float and current_dtype.is_float()
            )
            should_apply_repair = field.repair_with is not None and (
                field.repair_when_typed or expected_type is str or not already_target_numeric
            )

            if should_apply_repair:
                expr = field.repair_with(expr)

            if not should_apply_repair:
                if expected_type is int and current_dtype == pl.Int64:
                    continue
                if expected_type is float and current_dtype == pl.Float64:
                    continue
                if expected_type is str and current_dtype == pl.String:
                    continue

            if expected_type in (int, float) and current_dtype == pl.String:
                if expected_type is int:
                    direct_int = expr.cast(pl.Int64, strict=False)
                    as_float = expr.str.strip_chars().cast(pl.Float64, strict=False)
                    exact_float_int = (
                        pl.when(
                            as_float.is_not_null()
                            & (as_float == as_float.floor())
                            & as_float.is_finite()
                        )
                        .then(as_float.cast(pl.Int64, strict=False))
                        .otherwise(None)
                    )
                    expr = pl.coalesce([direct_int, exact_float_int])
                else:
                    normalized = SafeCoercer._normalize_numeric_string(expr)
                    expr = normalized.cast(pl.Float64, strict=False).fill_nan(None)

                coercion_exprs.append(expr.alias(col_name))

            elif expected_type is float and current_dtype.is_integer():
                coercion_exprs.append(expr.cast(pl.Float64).alias(col_name))

            elif expected_type is int and current_dtype.is_float():
                safe_int_expr = (
                    pl.when(expr.is_not_null() & (expr == expr.floor()) & expr.is_finite())
                    .then(expr.cast(pl.Int64, strict=False))
                    .otherwise(None)
                )
                coercion_exprs.append(safe_int_expr.alias(col_name))

            elif expected_type is str and current_dtype != pl.String:
                coercion_exprs.append(expr.cast(pl.String).alias(col_name))

            elif should_apply_repair:
                coercion_exprs.append(expr.alias(col_name))

        if not coercion_exprs:
            return df

        return lazy_df.with_columns(coercion_exprs).collect()
