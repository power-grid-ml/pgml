# Power Grid ML
Machine learning toolkit for state estimation, curriculum learning, and physical system modeling on power grids.

## Data Handling Concepts

The pipeline is architected to process massive simulation datasets that exceed system RAM. To maintain zero-copy data transfers and avoid the Python Global Interpreter Lock (GIL) bottlenecks, the following stack is employed:

1. **Analytical Extraction (DuckDB & Parquet):** 
   Relational databases (like PostgreSQL) are optimized for row-based writes, making bulk analytical reads extremely slow. DuckDB attaches directly to the PostgreSQL binary protocol to stream data out-of-core, sort it chronologically, and encode it directly into partitioned Apache Parquet files. This entirely bypasses standard Python ORM logic.
2. **Out-of-Core Aggregation (Polars LazyFrames):** 
   Finding dataset statistics (Mean, Standard Deviation, Min, Max) across gigabytes of simulation steps is required to normalize neural network inputs. Polars creates lazy query execution plans that stream Parquet row-groups from disk, compute the statistical moments iteratively, and release the memory. 
3. **Chunked Streaming (PyArrow & PyTorch):** 
   During training, PyTorch workers utilize PyArrow to load explicit chunk sizes of Parquet data. Vectorized logic slices these chunks into individual heterogeneous graph objects (`HeteroData`), minimizing latency and starving the GPU.

## Installation

For virtual environment management and deterministic dependency resolution, [pixi](https://pixi.prefix.dev/latest/installation/) is used. 

1. Install Pixi following [official documentation](https://pixi.prefix.dev/latest/installation/)
2. Clone the repository and navigate to the project root.
3. Install the environment dependencies defined in `pixi.toml`:
   ```bash
   pixi install
   ```
   To install CPU requirements only:
   ```bash
   pixi install -e cpu
   ```

## Run

Execution environments are explicitly split into CPU and GPU contexts to manage conflicting CUDA binaries. Use the `--environment` (short: `-e`) flag to specify the target hardware context.

Test the installations with the following commands:
```bash
pixi run --environment cpu python -c "import torch; print(torch.cuda.is_available())"
pixi run python -c "import torch; print(torch.cuda.is_available())"
```

To run the main training loop (after exporting the database to Parquet, and compiling stats):
```bash
pixi run python training/main.py
```
