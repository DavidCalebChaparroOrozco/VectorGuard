"""
VectorGuard Interactive Demo & Reproducible Comparison
Compares Native Python, Pandera, and VectorGuard across:
1. Execution speed (median over multiple runs after warm-up)
2. Peak memory footprint (tracemalloc)
3. Real-world resilience (dirty human-formatted data & row-level error isolation)
"""

import statistics
import time
import tracemalloc

import pandera as pa
import polars as pl
from pandera.polars import Check, Column, DataFrameSchema

from vectorguard import DataContract, Field, ValidationEngine


# ==============================================================================
# 1. DEFINE EQUIVALENT SCHEMAS
# ==============================================================================
class TransactionContract(DataContract):
    user_id: int = Field(gt=0, nullable=False)
    amount: float = Field(ge=0.0, le=5000.0, nullable=True)
    country: str = Field(length=2, regex="^[A-Z]{2}$", nullable=False)


pandera_schema = DataFrameSchema(
    {
        "user_id": Column(pl.Int64, Check.gt(0), nullable=False),
        "amount": Column(pl.Float64, Check.in_range(0.0, 5000.0), nullable=True),
        "country": Column(
            pl.String,
            [Check.str_length(2, 2), Check.str_matches("^[A-Z]{2}$")],
            nullable=False,
        ),
    }
)


# ==============================================================================
# 2. GENERATE REALISTIC DIRTY DATASET
# ==============================================================================
def build_dirty_dataset(n_rows: int = 500_000) -> pl.DataFrame:
    """Builds a dataset simulating real-world external CSV/Excel ingestion issues."""
    user_ids = [f" {i} " if i % 5 == 0 else str(i) for i in range(1, n_rows + 1)]
    amounts = [
        "N/A"
        if i % 20 == 0
        else f"{(i % 4500) + 10},5"
        if i % 3 == 0
        else str(float((i % 4500) + 10))
        for i in range(n_rows)
    ]
    countries = ["US" if i % 4 != 0 else "MX" for i in range(n_rows)]

    # Inject ~1% corrupt rows (invalid country code or negative amount)
    for idx in range(0, n_rows, 100):
        countries[idx] = "INVALID"
        amounts[idx] = "-50.0"

    return pl.DataFrame({"user_id": user_ids, "amount": amounts, "country": countries})


# ==============================================================================
# 3. IMPLEMENTATIONS TO COMPARE
# ==============================================================================
def run_native_python(df: pl.DataFrame) -> tuple[int, int]:
    """Row-by-row Python validation & manual cleaning."""
    valid_count = 0
    invalid_count = 0

    for row in df.iter_rows(named=True):
        try:
            uid = int(row["user_id"].strip())
            raw_amt = row["amount"].strip()
            if raw_amt in ("", "N/A", "NULL"):
                amt = None
            else:
                amt = float(raw_amt.replace(",", "."))

            country = row["country"]
            is_country_valid = len(country) == 2 and country.isalpha() and country.isupper()
            is_amt_valid = amt is None or (0.0 <= amt <= 5000.0)

            if uid > 0 and is_amt_valid and is_country_valid:
                valid_count += 1
            else:
                invalid_count += 1
        except (ValueError, AttributeError):
            invalid_count += 1

    return valid_count, invalid_count


def run_pandera_with_manual_cleaning(df: pl.DataFrame) -> tuple[int, int]:
    """
    Pandera requires manual pre-cleaning before validation and does not natively
    split clean vs. quarantined rows with per-row error traces in Polars.
    """
    cleaned = df.with_columns(
        [
            pl.col("user_id").str.strip_chars().cast(pl.Int64, strict=False),
            pl.when(pl.col("amount").str.strip_chars().is_in(["", "N/A", "NULL"]))
            .then(None)
            .otherwise(pl.col("amount").str.strip_chars().str.replace(",", ".", literal=True))
            .cast(pl.Float64, strict=False),
        ]
    )
    try:
        pandera_schema.validate(cleaned, lazy=True)
        return cleaned.height, 0
    except pa.errors.SchemaErrors as err:
        # Approximate failed rows from failure cases
        failed_indices = err.failure_cases["index"].drop_nulls().unique().len()
        return cleaned.height - failed_indices, failed_indices


def run_vectorguard(df: pl.DataFrame) -> tuple[int, int]:
    """VectorGuard: single-pass coercion, validation, and Dead Letter Box isolation."""
    clean_df, isolated_df = ValidationEngine.validate(df, TransactionContract, isolate=True)
    return clean_df.height, isolated_df.height


# ==============================================================================
# 4. BENCHMARK & DEMO RUNNER
# ==============================================================================
def profile_workload(name: str, fn, df: pl.DataFrame, repeats: int = 5):
    # Warm-up run
    valid_cnt, invalid_cnt = fn(df)

    times = []
    peak_mem_mb = 0.0

    for i in range(repeats):
        if i == 0:
            tracemalloc.start()
        t0 = time.perf_counter()
        fn(df)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if i == 0:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mem_mb = peak / (1024 * 1024)

    median_time = statistics.median(times)
    return {
        "name": name,
        "median_s": median_time,
        "peak_mem_mb": peak_mem_mb,
        "valid": valid_cnt,
        "invalid": invalid_cnt,
    }


if __name__ == "__main__":
    N_ROWS = 500_000
    print(f"Generating {N_ROWS:,} dirty records with human formatting errors...")
    dataset = build_dirty_dataset(N_ROWS)

    print("Running benchmarks (1 warm-up + 5 measured runs)...\n")
    results = [
        profile_workload("Native Python (Row Loop)", run_native_python, dataset),
        profile_workload("Pandera + Manual Cleaning", run_pandera_with_manual_cleaning, dataset),
        profile_workload("VectorGuard (Compiled Plan)", run_vectorguard, dataset),
    ]

    # Verify correctness equivalence across all engines
    expected_valid = results[-1]["valid"]
    expected_invalid = results[-1]["invalid"]
    for res in results:
        assert res["valid"] == expected_valid, f"Mismatch in {res['name']}"
        assert res["invalid"] == expected_invalid, f"Mismatch in {res['name']}"

    vg_time = results[-1]["median_s"]

    print(f"{'Engine':<32} | {'Median Time':<12} | {'Relative':<16} | {'Python Peak Mem':<15}")
    print("-" * 84)
    for res in results:
        ratio = res["median_s"] / vg_time
        if res["name"].startswith("VectorGuard"):
            rel = "1.0x (Baseline)"
        elif ratio >= 1.0:
            rel = f"{ratio:.1f}x slower"
        else:
            rel = f"{1.0 / ratio:.1f}x faster"
        print(
            f"{res['name']:<32} | {res['median_s']:>9.4f} s | {rel:<16} | {res['peak_mem_mb']:>11.2f} MB"
        )

    print("\n--- VectorGuard Dead Letter Box Preview (First 3 Quarantined Rows) ---")
    _, quarantined = ValidationEngine.validate(dataset, TransactionContract, isolate=True)
    print(quarantined.head(3))
