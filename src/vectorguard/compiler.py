from dataclasses import dataclass

import polars as pl

from vectorguard.coercion import SafeCoercer
from vectorguard.contract import DataContract
from vectorguard.fields import Field

ERROR_COLUMN_NAME = "_vg_errors"


@dataclass(frozen=True)
class CompiledRule:
    """Represents a single compiled vectorized validation check."""

    col_name: str
    rule_name: str
    check_col_name: str
    expr: pl.Expr


@dataclass(frozen=True)
class CompiledContract:
    """
    Pre-compiled execution plan for a DataContract.
    Separates schema rule compilation from runtime DataFrame execution.
    """

    contract_name: str
    fields: dict[str, Field]
    rules: tuple[CompiledRule, ...]

    @classmethod
    def compile(cls, contract: type[DataContract]) -> "CompiledContract":
        fields = contract.get_fields()
        rules: list[CompiledRule] = []

        for col_name, field in fields.items():
            if not field.nullable:
                rules.append(
                    CompiledRule(
                        col_name=col_name,
                        rule_name="null",
                        check_col_name=f"__vg_check_{col_name}_null",
                        expr=pl.col(col_name).is_null(),
                    )
                )

            if field.gt is not None:
                rules.append(
                    CompiledRule(
                        col_name=col_name,
                        rule_name="gt",
                        check_col_name=f"__vg_check_{col_name}_gt",
                        expr=(pl.col(col_name) <= field.gt).fill_null(False),
                    )
                )

            if field.ge is not None:
                rules.append(
                    CompiledRule(
                        col_name=col_name,
                        rule_name="ge",
                        check_col_name=f"__vg_check_{col_name}_ge",
                        expr=(pl.col(col_name) < field.ge).fill_null(False),
                    )
                )

            if field.lt is not None:
                rules.append(
                    CompiledRule(
                        col_name=col_name,
                        rule_name="lt",
                        check_col_name=f"__vg_check_{col_name}_lt",
                        expr=(pl.col(col_name) >= field.lt).fill_null(False),
                    )
                )

            if field.le is not None:
                rules.append(
                    CompiledRule(
                        col_name=col_name,
                        rule_name="le",
                        check_col_name=f"__vg_check_{col_name}_le",
                        expr=(pl.col(col_name) > field.le).fill_null(False),
                    )
                )

            if field.length is not None:
                rules.append(
                    CompiledRule(
                        col_name=col_name,
                        rule_name="length",
                        check_col_name=f"__vg_check_{col_name}_length",
                        expr=(pl.col(col_name).str.len_chars() != field.length).fill_null(False),
                    )
                )

            if field.regex is not None:
                rules.append(
                    CompiledRule(
                        col_name=col_name,
                        rule_name="regex",
                        check_col_name=f"__vg_check_{col_name}_regex",
                        expr=(pl.col(col_name).str.contains(field.regex).not_()).fill_null(False),
                    )
                )

        return cls(
            contract_name=contract.__name__,
            fields=fields,
            rules=tuple(rules),
        )

    def build_coercion_exprs(self, schema: dict[str, pl.DataType]) -> list[pl.Expr]:
        """Builds vectorized coercion expressions tailored to the input schema."""
        coercion_exprs: list[pl.Expr] = []

        for col_name, field in self.fields.items():
            if col_name not in schema:
                continue

            expected_type = field.data_type
            current_dtype = schema[col_name]
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

        return coercion_exprs
