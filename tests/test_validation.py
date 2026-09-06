import pytest
import polars as pl
from vectorguard import DataContract, Field, ValidationEngine, ValidationError
from vectorguard import validate_contract

# 1. Define a sample data contract for testing
class TransactionContract(DataContract):
    user_id: int = Field(gt=0)
    amount: float = Field(ge=0.0, le=5000.0)
    country: str = Field(length=2, regex="^[A-Z]{2}$")

# Define a clean contract for testing decorator integrations
class PipelineContract(DataContract):
    step_id: int = Field(gt=0)
    status: str = Field(length=2)

# 2. Test case: Valid data should pass seamlessly
def test_valid_dataframe_passes():
    # Explicitly creating list variables to bypass any parser trimming
    valid_ids = [1, 2, 3, 4]
    valid_amounts = [10.5, 99.9, 0.0, 4999.9]
    valid_countries = ["US", "MX", "CA", "FR"]
    
    valid_data = {
        "user_id": valid_ids,
        "amount": valid_amounts,
        "country": valid_countries
    }
    df = pl.DataFrame(valid_data)
    
    # Act & Assert
    result = ValidationEngine.validate(df, TransactionContract)
    assert result.shape == (4, 3)  # Should return the original dataframe intact

# 3. Test case: Invalid data should raise ValidationError with accurate error counts
def test_invalid_dataframe_raises_errors():
    invalid_ids = [0, 1, 2]
    invalid_amounts = [-5.0, 150.0, 6000.0]
    invalid_countries = ["US", "MEX", "ca"]
    
    invalid_data = {
        "user_id": invalid_ids,          # 0 fails gt=0
        "amount": invalid_amounts,       # -5.0 fails ge=0, 6000.0 fails le=5000
        "country": invalid_countries     # 'MEX' fails length=2, 'ca' fails regex pattern
    }
    df = pl.DataFrame(invalid_data)
    
    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        ValidationEngine.validate(df, TransactionContract)
    
    errors = exc_info.value.errors
    
    # Verify the error dictionary structure and counts
    assert "user_id" in errors
    assert "gt" in errors["user_id"]
    
    assert "amount" in errors
    assert "ge" in errors["amount"]
    assert "le" in errors["amount"]
    
    assert "country" in errors
    assert "length" in errors["country"]
    assert "regex" in errors["country"]

# 4. Test case: Safe coercion repairs dirty data
def test_safe_coercion_repairs_dirty_data():
    # Simulate an external file manipulated by manual data entry issues
    dirty_data = {
        "user_id": ["1", "2", " 3 ", "4"],
        "amount": ["10,5", "99.9", "N/A", "4999,0"],
        "country": ["US", "MX", "CA", "FR"]
    }
    df = pl.DataFrame(dirty_data)
    
    class CoercionContract(DataContract):
        user_id: int = Field(gt=0, nullable=False)
        amount: float = Field(ge=0.0, le=5000.0, nullable=True)
        country: str = Field(length=2)

    # Act
    repaired_df = ValidationEngine.validate(df, CoercionContract)
    
    # Assert structural type mutations succeeded
    assert repaired_df.schema["user_id"] == pl.Int64
    assert repaired_df.schema["amount"] == pl.Float64
    
    # Assert accurate text parsing occurred
    assert repaired_df["user_id"][2] == 3        # " 3 " -> 3
    assert repaired_df["amount"][0] == 10.5      # "10,5" -> 10.5
    assert repaired_df["amount"][2] is None      # "N/A" -> native null

# 5. Test case: Decorator automatically validates and mutates incoming DataFrames
# Wrap a mock transformation pipeline step with our decorator
@validate_contract(contract=PipelineContract)
def run_pipeline_step(df: pl.DataFrame, factor: int = 2) -> pl.DataFrame:
    """
    A mock transformation pipeline step that multiplies the 'step_id' column by a given factor.
    This function is decorated with @validate_contract to automatically validate and coerce
    the incoming DataFrame against the PipelineContract before processing.
    """
    return df.with_columns(pl.col("step_id") * factor)

def test_decorator_automatically_validates_and_mutates():
    # Setup dirty data that requires safe coercion first
    raw_data = {
        "step_id": ["10", " 20 ", "30"],  # Str inputs that need auto-casting to int
        "status": ["OK", "OK", "OK"]
    }
    df = pl.DataFrame(raw_data)

    # Act
    output_df = run_pipeline_step(df, factor=10)

    # Assert: Verify coercion succeeded, validation passed, and processing completed
    assert output_df.schema["step_id"] == pl.Int64
    assert output_df["step_id"].to_list() == [100, 200, 300]

# 6. Test case: Dead letter box isolates corrupt rows
def test_dead_letter_box_isolates_corrupt_rows():
    # Setup variables explicitly to avoid syntax trimming errors
    mixed_data = {
        "user_id": [0, 10, 30],         # 0 breaks gt=0
        "amount": [150.0, 20.0, -5.0],   # -5.0 breaks ge=0
        "country": ["US", "XX", "CA"]
    }
    df = pl.DataFrame(mixed_data)

    class IsolationContract(DataContract):
        user_id: int = Field(gt=0)
        amount: float = Field(ge=0.0)
        country: str = Field(length=2)

    # Act: Trigger validation with isolation mode enabled
    clean_df, dead_letter_df = ValidationEngine.validate(df, IsolationContract, isolate=True)

    # Assert: Verify clean rows were retained correctly
    assert clean_df.shape == (1, 3)
    assert clean_df["user_id"].to_list() == [10]

    # Assert: Verify flawed rows were diverted into the dead letter box
    assert dead_letter_df.shape == (2, 4)
    assert dead_letter_df["user_id"].to_list() == [0, 30]

# 7. Test custom repair & error reason logging
def test_custom_repair_and_error_reason_logging():
    # Row 0: Needs custom string stripping ("ID_99") -> valid
    # Row 1: Breaks amount range (gt=0) -> invalid
    # Row 2: Breaks country length -> invalid
    test_data = {
        "user_id": ["ID_99", "ID_100", "ID_200"],
        "amount": [50.0, -10.0, 20.0],
        "country": ["US", "MX", "BAD_NAME"]
    }
    df = pl.DataFrame(test_data)

    class CustomContract(DataContract):
        # A custom repair rule that removes the "ID_" prefix from user string entry via polars expression
        user_id: int = Field(gt=0, repair_with=lambda expr: expr.str.replace("ID_", "", literal=True))
        amount: float = Field(gt=0)
        country: str = Field(length=2)

    # Act
    clean_df, dead_letter_df = ValidationEngine.validate(df, CustomContract, isolate=True)

    # Assert: Custom repair worked on valid rows
    assert clean_df.shape == (1, 3)
    assert clean_df["user_id"][0] == 99  # "ID_99" -> "99" -> 99 Int

    # Assert: Explanatory error tracking logs exist
    assert dead_letter_df.shape == (2, 4)  # 3 original columns + 1 error log column
    assert "_vg_errors" in dead_letter_df.columns
    
    # Check that it explains exactly WHY it failed
    assert "[amount_gt]" in dead_letter_df["_vg_errors"][0]
    assert "[country_length]" in dead_letter_df["_vg_errors"][1]

# 8. Test case: Multiple validation failures are tracked on the same row
def test_multiple_error_reasons_are_logged():
    # Row 0 violates amount.gt, country.length, and country.regex constraints.
    multi_ids = [10]
    multi_amounts = [-100.0]
    multi_countries = ["BAD"]

    test_data = {
        "user_id": multi_ids,
        "amount": multi_amounts,
        "country": multi_countries
    }
    df = pl.DataFrame(test_data)

    class MultiErrorContract(DataContract):
        user_id: int = Field(gt=0)
        amount: float = Field(gt=0)
        country: str = Field(length=2, regex="^[A-Z]{2}$")

    # Act
    clean_df, dead_letter_df = ValidationEngine.validate(
        df,
        MultiErrorContract,
        isolate=True
    )

    # Assert: The row is isolated because multiple rules failed
    assert clean_df.shape == (0, 3)
    assert dead_letter_df.shape == (1, 4)

    # Assert: All triggered validation rules are preserved in the trace
    error_trace = dead_letter_df["_vg_errors"][0]

    assert "[amount_gt]" in error_trace
    assert "[country_length]" in error_trace
    assert "[country_regex]" in error_trace

    # Ensure the trace contains multiple rule markers rather than a single reason.
    assert error_trace.count("[") == 3


# 9. Test case: Nullable fields accept native null values
def test_nullable_field_accepts_null():
    data = {
        "user_id": [1, 2, 3],
        "amount": [10.0, None, 50.0],
        "country": ["US", "MX", "CA"]
    }
    df = pl.DataFrame(data)

    class NullableContract(DataContract):
        user_id: int = Field(gt=0)
        amount: float = Field(ge=0.0, nullable=True)
        country: str = Field(length=2)

    # Act
    result = ValidationEngine.validate(df, NullableContract)

    # Assert: Nullable null values should not trigger a validation failure
    assert result.shape == (3, 3)
    assert result["amount"][1] is None


# 10. Test case: Non-nullable fields reject native null values
def test_non_nullable_field_rejects_null():
    data = {
        "user_id": [1, None, 3],
        "amount": [10.0, 20.0, 50.0],
        "country": ["US", "MX", "CA"]
    }
    df = pl.DataFrame(data)

    class NonNullableContract(DataContract):
        user_id: int = Field(gt=0, nullable=False)
        amount: float = Field(ge=0.0)
        country: str = Field(length=2)

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        ValidationEngine.validate(df, NonNullableContract)

    errors = exc_info.value.errors

    assert "user_id" in errors
    assert "null" in errors["user_id"]
    assert errors["user_id"]["null"] == "Failed 1 rows"


# 11. Test case: Structural validation rejects missing required columns
def test_missing_required_columns_raise_validation_error():
    data = {
        "user_id": [1, 2, 3],
        "amount": [10.0, 20.0, 30.0]
    }
    df = pl.DataFrame(data)

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        ValidationEngine.validate(df, TransactionContract)

    errors = exc_info.value.errors

    assert "structure" in errors
    assert "country" in errors["structure"]


# 12. Test case: Contracts without validation rules pass unchanged
def test_contract_without_validation_rules_passes():
    data = {
        "user_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    }
    df = pl.DataFrame(data)

    class NoRulesContract(DataContract):
        user_id: int = Field()
        name: str = Field()

    # Act
    result = ValidationEngine.validate(df, NoRulesContract)

    # Assert
    assert result.shape == (3, 2)
    assert result.columns == ["user_id", "name"]


# 13. Test case: Isolation mode with no validation rules returns an empty dead letter box
def test_isolation_without_validation_rules_returns_empty_dead_letter_box():
    data = {
        "user_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    }
    df = pl.DataFrame(data)

    class NoRulesContract(DataContract):
        user_id: int = Field()
        name: str = Field()

    # Act
    clean_df, dead_letter_df = ValidationEngine.validate(
        df,
        NoRulesContract,
        isolate=True
    )

    # Assert: All rows are valid
    assert clean_df.shape == (3, 2)

    # Assert: Dead letter box is empty but still exposes the tracing schema
    assert dead_letter_df.shape == (0, 3)
    assert "_vg_errors" in dead_letter_df.columns
    assert dead_letter_df["_vg_errors"].to_list() == []


# 14. Test case: Custom repair is applied before standard numeric coercion
def test_custom_repair_runs_before_standard_coercion():
    data = {
        "user_id": ["ID_10", "ID_20", "ID_30"],
        "amount": ["10,5", "20,5", "30,5"],
        "country": ["US", "MX", "CA"]
    }
    df = pl.DataFrame(data)

    class CombinedContract(DataContract):
        user_id: int = Field(
            gt=0,
            repair_with=lambda expr: expr.str.replace(
                "ID_",
                "",
                literal=True
            )
        )
        amount: float = Field(ge=0.0)
        country: str = Field(length=2)

    # Act
    result = ValidationEngine.validate(df, CombinedContract)

    # Assert: Custom repair and standard coercion both executed successfully
    assert result.schema["user_id"] == pl.Int64
    assert result.schema["amount"] == pl.Float64

    assert result["user_id"].to_list() == [10, 20, 30]
    assert result["amount"].to_list() == [10.5, 20.5, 30.5]


# 15. Test case: Valid isolated output never contains the internal error column
def test_clean_dataframe_does_not_expose_error_trace():
    data = {
        "user_id": [10, 20],
        "amount": [100.0, 200.0],
        "country": ["US", "MX"]
    }
    df = pl.DataFrame(data)

    class IsolationContract(DataContract):
        user_id: int = Field(gt=0)
        amount: float = Field(ge=0.0)
        country: str = Field(length=2)

    # Act
    clean_df, dead_letter_df = ValidationEngine.validate(
        df,
        IsolationContract,
        isolate=True
    )

    # Assert: Clean output preserves the public DataFrame schema
    assert "_vg_errors" not in clean_df.columns

    # Assert: Dead letter output retains the diagnostic information
    assert "_vg_errors" in dead_letter_df.columns
    assert clean_df.shape == (2, 3)
    assert dead_letter_df.shape == (0, 4)