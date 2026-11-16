from typing import Any

import polars as pl

from vectorguard.compiler import ERROR_COLUMN_NAME, CompiledContract
from vectorguard.contract import DataContract

_COMPILED_CACHE: dict[type[DataContract], CompiledContract] = {}


class ValidationError(Exception):
    """Exception raised when a DataFrame fails contract compliance."""

    def __init__(self, errors: dict[str, Any]):
        self.errors = errors
        super().__init__(f"Contract validation failed: {errors}")


class ValidationEngine:
    @staticmethod
    def get_compiled_contract(contract: type[DataContract]) -> CompiledContract:
        """Retrieves or compiles and caches a DataContract execution plan."""
        compiled = _COMPILED_CACHE.get(contract)
        if compiled is None:
            compiled = CompiledContract.compile(contract)
            _COMPILED_CACHE[contract] = compiled
        return compiled

    @staticmethod
    def validate(
        df: pl.DataFrame,
        contract: type[DataContract],
        isolate: bool = False,
    ) -> pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame]:
        """
        Validates a Polars DataFrame against a DataContract using a unified lazy plan.

        If isolate=False (default): Raises ValidationError on failure.
        If isolate=True: Returns a tuple of (clean_df, isolated_df) without crashing.
        """
        compiled = ValidationEngine.get_compiled_contract(contract)

        # 1. Structural Validation
        missing_cols = [col for col in compiled.fields if col not in df.columns]
        if missing_cols:
            raise ValidationError({"structure": f"Missing required columns: {missing_cols}"})

        if isolate and ERROR_COLUMN_NAME in df.columns:
            raise ValidationError(
                {
                    "structure": (
                        f"Input DataFrame already contains reserved diagnostic column "
                        f"'{ERROR_COLUMN_NAME}'. Rename or drop it before running isolate=True."
                    )
                }
            )

        # 2. Build unified LazyFrame pipeline (Coercion + Validation in a single query plan)
        lazy_plan = df.lazy()
        coercion_exprs = compiled.build_coercion_exprs(dict(df.schema))
        if coercion_exprs:
            lazy_plan = lazy_plan.with_columns(coercion_exprs)

        # If there are no validation rules, materialize coerced DataFrame once
        if not compiled.rules:
            coerced_df = lazy_plan.collect()
            if isolate:
                empty_isolated = coerced_df.with_columns(
                    pl.lit("").alias(ERROR_COLUMN_NAME)
                ).filter(pl.lit(False))
                return coerced_df, empty_isolated
            return coerced_df

        # 3. Single-pass materialization for both coercion and rule evaluation
        rule_exprs = [rule.expr.alias(rule.check_col_name) for rule in compiled.rules]

        if isolate:
            trace_parts = [
                pl.when(pl.col(rule.check_col_name))
                .then(pl.lit(f"[{rule.col_name}_{rule.rule_name}]"))
                .otherwise(pl.lit(""))
                for rule in compiled.rules
            ]
            invalid_mask_col = "__vg_is_invalid__"

            evaluated_df = (
                lazy_plan.with_columns(rule_exprs)
                .with_columns(
                    [
                        pl.any_horizontal([pl.col(r.check_col_name) for r in compiled.rules]).alias(
                            invalid_mask_col
                        ),
                        pl.concat_str(trace_parts, separator=" ")
                        .str.strip_chars()
                        .alias(ERROR_COLUMN_NAME),
                    ]
                )
                .drop([r.check_col_name for r in compiled.rules])
                .collect()
            )

            is_invalid = evaluated_df[invalid_mask_col]
            clean_df = evaluated_df.filter(~is_invalid).drop([invalid_mask_col, ERROR_COLUMN_NAME])
            isolated_df = evaluated_df.filter(is_invalid).drop(invalid_mask_col)
            return clean_df, isolated_df

        # Strict mode: materialize coerced columns + boolean check flags in a single pass
        evaluated_df = lazy_plan.with_columns(rule_exprs).collect()
        check_cols = [r.check_col_name for r in compiled.rules]

        has_any_failure = evaluated_df.select(pl.any_horizontal(check_cols).any()).item()
        if has_any_failure:
            errors: dict[str, dict[str, str]] = {}
            for rule in compiled.rules:
                fail_count = evaluated_df[rule.check_col_name].sum()
                if fail_count > 0:
                    errors.setdefault(rule.col_name, {})[rule.rule_name] = (
                        f"Failed {fail_count} rows"
                    )
            raise ValidationError(errors)

        return evaluated_df.drop(check_cols)
