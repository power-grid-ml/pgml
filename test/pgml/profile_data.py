
import torch
import time
from pathlib import Path
from pgml.data_pipeline.dataloader import get_dataloader

def profile_dataloader():
    input_dir = Path("./data/input")
    if not input_dir.exists():
        print(f"Input dir {input_dir} does not exist. Skipping profiling.")
        return

    train_loader = get_dataloader(
        base_data_dir=input_dir,
        dataset_ids=[1],
        batch_size=1,
        num_workers=0,
        chunk_size_rows=50000,
    )

    print("Starting profiling...")
    start_time = time.time()
    num_steps = 10
    for i, batch in enumerate(train_loader):
        if i >= num_steps:
            break
        print(f"Step {i} loaded")
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_steps
    print(f"Average time per step (batch_size=1, num_workers=0): {avg_time:.4f}s")

if __name__ == "__main__":
    profile_dataloader()
