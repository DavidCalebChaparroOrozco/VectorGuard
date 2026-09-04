import functools
import inspect
import polars as pl
from typing import Type, Callable, Any
from vectorguard.contract import DataContract
from vectorguard.core import ValidationEngine

def validate_contract(contract: Type[DataContract]) -> Callable:
    """
    A decorator to automatically validate and coerce incoming Polars DataFrames
    or LazyFrames against a specific DataContract before function execution.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Inspect function signature to map arguments correctly
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Bind args and kwargs to the signature to find where the DataFrame is located
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Iterate over arguments to locate and validate Polars frames
            for param_name, param_value in bound_args.arguments.items():
                if isinstance(param_value, pl.DataFrame):
                    # Validate and replace with the safe, coerced DataFrame
                    validated_df = ValidationEngine.validate(param_value, contract)
                    bound_args.arguments[param_name] = validated_df
                
                elif isinstance(param_value, pl.LazyFrame):
                    # If it's a LazyFrame, we collect it to validate, then turn it back into a LazyFrame
                    # This ensures contract enforcement while keeping pipeline safety intact
                    collected_df = param_value.collect()
                    validated_df = ValidationEngine.validate(collected_df, contract)
                    bound_args.arguments[param_name] = validated_df.lazy()

            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator
