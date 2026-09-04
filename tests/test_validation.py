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
    # Setup mixed input where rows 0 and 2 are perfectly fine, but row 1 is broken
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
    assert dead_letter_df.shape == (2, 3)
    assert dead_letter_df["user_id"].to_list() == [0, 30]
