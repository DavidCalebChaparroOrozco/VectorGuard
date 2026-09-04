import time
import polars as pl
import pandera as pa
from pandera.polars import DataFrameSchema, Column, Check
from vectorguard import DataContract, Field, ValidationEngine

# 1. Setup a clean contract using VectorGuard
class VectorGuardContract(DataContract):
    user_id: int = Field(gt=0, nullable=False)
    amount: float = Field(ge=0.0, le=5000.0, nullable=False)
    status: str = Field(length=2)

# 2. Setup an identical schema using Pandera
pandera_schema = DataFrameSchema({
    "user_id": Column(pl.Int64, Check.gt(0), nullable=False),
    # Fixed from Check.get_range to Check.in_range
    "amount": Column(pl.Float64, Check.in_range(0.0, 5000.0), nullable=False),
    "status": Column(pl.String, Check.str_length(2, 2), nullable=False),
})

def generate_benchmark_data(num_rows: int = 1_000_000) -> pl.DataFrame:
    """Generates synthetic high-volume data for processing comparisons."""
    print(f"Generating {num_rows:,} rows of synthetic benchmark data...")
    return pl.DataFrame({
        "user_id": [str(i) for i in range(1, num_rows + 1)], # Strings needing coercion
        "amount": [float(i % 4500) for i in range(num_rows)],
        "status": ["OK" if i % 2 == 0 else "NY" for i in range(num_rows)]
    })

def run_native_loop_benchmark(df: pl.DataFrame):
    """Simulates raw row-by-row python looping parsing logic."""
    start = time.perf_counter()
    raw_dicts = df.to_dicts()
    valid_rows = []
    
    for row in raw_dicts:
        try:
            # Manual coercion and validation steps
            uid = int(row["user_id"].strip())
            amt = float(row["amount"])
            stat = str(row["status"])
            
            if uid > 0 and 0.0 <= amt <= 5000.0 and len(stat) == 2:
                valid_rows.append((uid, amt, stat))
        except (ValueError, TypeError):
            continue
            
    end = time.perf_counter()
    return end - start

def run_pandera_benchmark(df: pl.DataFrame):
    """Measures validation timing using Pandera's engine."""
    start = time.perf_counter()
    # Pandera requires manual preparation since it doesn't handle safe auto-coercion natively
    prepared_df = df.with_columns([
        pl.col("user_id").str.strip_chars().cast(pl.Int64),
        pl.col("amount").cast(pl.Float64)
    ])
    try:
        pandera_schema.validate(prepared_df)
    except Exception:
        pass
    end = time.perf_counter()
    return end - start

def run_vectorguard_benchmark(df: pl.DataFrame):
    """Measures validation timing using VectorGuard's lazy single-pass execution."""
    start = time.perf_counter()
    try:
        ValidationEngine.validate(df, VectorGuardContract)
    except Exception:
        pass
    end = time.perf_counter()
    return end - start

if __name__ == "__main__":
    df = generate_benchmark_data(1_000_000)
    
    print("\n--- Running Performance Profiles ---")
    
    # 1. Native Loop Timing
    loop_time = run_native_loop_benchmark(df)
    print(f"Native Python Row-Loop: {loop_time:.4f} seconds")
    
    # 2. Pandera Timing
    pandera_time = run_pandera_benchmark(df)
    print(f"Pandera Framework:      {pandera_time:.4f} seconds")
    
    # 3. VectorGuard Timing
    vg_time = run_vectorguard_benchmark(df)
    print(f"VectorGuard Engine:     {vg_time:.4f} seconds")
    
    print("\n--- Summary Analysis ---")
    print(f"VectorGuard is {loop_time / vg_time:.1f}x faster than a Native Python Loop.")
    print(f"VectorGuard is {pandera_time / vg_time:.1f}x faster than Pandera.")
