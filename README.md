# 🛡️ VectorGuard

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Engine: Polars](https://img.shields.io/badge/Engine-Polars-orange.svg)](https://pola.rs/)

**VectorGuard** is a high-performance, declarative data contract validation and sanitization library for Python data pipelines.

Built on top of **Polars' expression engine**, VectorGuard translates schema definitions and validation constraints into vectorized operations, minimizing Python-level execution overhead and leveraging native, multi-threaded processing.

Designed for production-oriented data workflows, it combines schema enforcement, type coercion, data normalization, and row-level error isolation into a unified validation layer.

---

## 🚀 Core Capabilities

- **Vectorized Validation Engine:** Compiles declarative field constraints into native Polars expressions, avoiding conventional row-by-row Python validation and reducing execution overhead.

- **Automated Data Normalization & Coercion:** Handles common data quality inconsistencies, including whitespace normalization, string-to-numeric conversion, localized decimal separators, and configurable null-value representations.

- **Row-Level Error Isolation:** Implements a Dead Letter Queue (DLQ) pattern to separate invalid records from successfully validated data, enabling resilient pipelines without discarding entire batches.

- **Declarative Data Contracts:** Defines schemas, types, nullability, and business constraints through reusable Python models, separating validation logic from data processing operations.

- **Pipeline Integration:** Provides a decorator-based API that integrates contract validation directly into existing Polars processing workflows.

- **Native Execution Optimization:** Delegates computationally intensive operations to Polars' execution engine, benefiting from columnar processing and parallel execution.

---

## ⚡ Performance Benchmarks

VectorGuard is designed to minimize validation overhead in high-volume data processing workloads.

The following benchmark evaluates validation and sanitization performance across **1,000,000 records**, including inconsistent numeric representations, trailing whitespace, and boundary constraint checks.

| Framework / Engine | Execution Time | Relative Performance |
|:---|---:|---:|
| **VectorGuard** | **0.1527 s** | **1.0x** |
| Pandera | 0.3279 s | 2.1x slower |
| Native Python Row-Based Validation | 1.6846 s | 11.0x slower |

**Benchmark highlights:**
- Approximately **2.1x faster** than Pandera in the tested workload.
- Approximately **11x faster** than conventional Python row-based validation.
- Processes one million records in approximately 153 milliseconds under the tested environment.

<sub>Benchmark results are workload- and environment-dependent. Measurements were obtained in a GitHub Codespaces Linux environment. Reproducibility requires matching dataset characteristics, validation rules, library versions, and execution conditions.</sub>

---

## 📦 Installation

Install VectorGuard using `uv` or `pip`:

```bash
# Using uv
uv pip install vectorguard

# Using pip
pip install vectorguard
```

---

## 💻 Quick Start

### 1. Define a Declarative Data Contract

Define the expected schema and business constraints using reusable contract definitions.

```python
from vectorguard import DataContract, Field


class TransactionContract(DataContract):
    user_id: int = Field(gt=0, nullable=False)
    amount: float = Field(ge=0.0, le=5000.0, nullable=True)
    country: str = Field(length=2, regex="^[A-Z]{2}$")
```

### 2. Integrate Validation into Data Pipelines

Use the `@validate_contract` decorator to enforce data contracts at pipeline boundaries.

```python
import polars as pl
from vectorguard import validate_contract


@validate_contract(contract=TransactionContract)
def process_batch(df: pl.DataFrame) -> pl.DataFrame:
    # Data is validated and normalized before processing.
    return df.filter(pl.col("amount") > 100.0)


dirty_data = pl.DataFrame(
    {"user_id": ["1", " 2 ", "3"], "amount": ["10,5", "99.9", "N/A"], "country": ["US", "MX", "CA"]}
)

clean_df = process_batch(dirty_data)
```

VectorGuard applies the configured normalization and coercion rules before enforcing the contract, allowing downstream transformations to operate on a consistent data representation.

### 3. Isolate Invalid Records

For workflows that require fault tolerance and data lineage, VectorGuard supports row-level error isolation through its validation engine.

```python
from vectorguard import ValidationEngine

clean_df, isolated_df = ValidationEngine.validate(dirty_data, TransactionContract, isolate=True)
```

The resulting DataFrames serve distinct purposes:

- **`clean_df`:** Records that successfully satisfy the configured data contract, ready for downstream transformations and storage.
- **`isolated_df`:** Records that fail validation, available for auditing, remediation, and integration with external Dead Letter Queue (DLQ) systems.

This approach enables resilient batch processing while preserving invalid records for investigation instead of silently discarding them.

---

## 🏗️ Design Philosophy

VectorGuard follows three core engineering principles:

1. **Performance by Design:** Leverage columnar execution and vectorized expressions instead of Python-level iteration.
2. **Explicit Data Quality:** Make schema expectations and business constraints declarative, reusable, and enforceable.
3. **Resilient Data Processing:** Isolate invalid records without compromising the processing of otherwise valid batches.

The goal is to make data quality enforcement a first-class component of the data engineering lifecycle rather than an isolated validation step.

---

## 🛡️ License

VectorGuard is distributed under the **MIT License**. See the [LICENSE](LICENSE) file for details.