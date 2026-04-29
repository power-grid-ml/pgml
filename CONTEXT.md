# DESIGN.md: Power Grid Multimodal State Estimation

## 1. Purpose & Vision
This repository implements a **hierarchical, multimodal machine learning architecture** for power-system state estimation. 

**Ultimate Goal:** Identify and attribute power-quality disturbances (e.g., harmonic distortions) to specific devices under partial grid observability.

**Multimodal Vision:** The architecture uses a unified latent space. In the future, this allows the model to ingest both frequency-domain (harmonics) and time-domain (oscilloscope/wavelet) signals, fuse them across a physical grid topology, and reconstruct missing waveforms anywhere in the grid.

---

## 2. Core Strategy & Pipeline

The pipeline operates in three distinct levels: **Token** $\to$ **Entity** $\to$ **Graph**.

1.  **Tokenization (Data Layer):** Raw simulation steps are loaded from chunked parquet files. Measurements are converted into padded sets of tokens (Real/Imaginary complex values).
2.  **Local Encoding (Entity Layer):** Local neural networks encode variable-length token sets into fixed-size latent vectors (`hidden_dim`) for Nodes, Edges, and Devices independently.
3.  **Graph Inference (Graph Layer):** Devices are explicitly fused into their attached nodes. A Graph Neural Network (GNN) propagates information across the grid topology to infer missing states.
4.  **Decoding:** The updated latent vectors are decoded back into physical values (voltages, currents, spectra).

---

## 3. Data Representation (`HeteroData`)

One PyG `HeteroData` object corresponds to **one full simulation step**.

*   **Node Store (`graph["node"]`):** Static features, Measurement Tokens, Target Tokens.
*   **Edge Store (`graph[("node", "physical", "node")]`):** Static features, Measurement Tokens, Target Tokens.
*   **Device Store (`graph["device"]`):** Static features, Device Type, Parameter Tokens, Spectrum Tokens.
*   **Device Attachment (`graph[("device", "attached_to", "node")]`):** Bipartite edge mapping preserving explicit device identity without aggregating them into node features.

---

## 4. Software Architecture

### Data Pipeline (`pgml/data_pipeline/`)
*   `step_graph_stream.py` / `multi_table_step_stream.py`: Streams parquet rows in out-of-core chunks to prevent memory blowups.
*   `graph_assembler.py`: Assembles aligned parquet rows into PyG `HeteroData` objects.
*   `tokenizer.py`: Converts complex electrical values into PyTorch token batches.

### Model Stack (`pgml/models/`)
*   `token_encoders.py`: Local MLPs with Masked Mean Pooling converting tokens to latent vectors.
*   `masking.py`: Applies Latent Observability Masking (Nodes/Edges) and Pseudo-Measurement Noising (Devices).
*   `fusion.py`: Merges Node Latents, Static Features, and pooled Device Latents.
*   `graph_state_estimator.py`: `TransformerConv` GNN.
*   `decoders.py`: `TokenConditionedDecoder` that predicts harmonic-specific values based on target frequency and type embeddings.
*   `multimodal_state_estimator.py`: The orchestrator class wiring Encoders $\to$ Masking $\to$ Fusion $\to$ GNN $\to$ Decoders.

### Training (`pgml/training/`)
*   `curriculum.py`: Defines `TrainingStage` schedules (Bypass GNN, Mask Ratios, LRs).
*   `multitask_engine.py`: Lightning Module handling the loss calculations and optimizer groups.
*   `evaluation.py`: Computes device-type and frequency-specific validation reports.

---

## 5. Training Curriculum (Staged Training)

Training is controlled by a `TrainingCurriculum` consisting of `TrainingStage`s. 

1.  **Autoencoder Pretraining (`bypass_gnn=True`):** The GNN is skipped. Local encoders and decoders learn to map raw signals to a stable latent space and back. Masking is 0%.
2.  **Masked Autoencoder (`bypass_gnn=True`):** Node/Edge masks and Device noise are slowly introduced to teach the encoders robust representations.
3.  **Graph Transition (`bypass_gnn=False`):** The GNN is unfrozen. The model learns to use grid topology to correct the masked/noisy local encodings.
4.  **Harsh Observability:** Masking reaches realistic levels (e.g., 90% unmeasured nodes). The model performs true state estimation.

---

## 6. Current Simplifications & Technical Debt
*   *Time-Domain Signals:* Not yet implemented. Waiting for CWT-ViT encoders.
*   *Device Pooling:* Node-Device fusion uses simple sum/mean pooling. Should be upgraded to Attention.
*   *Decoders:* `TokenConditionedDecoder` uses `.expand()`. A true Cross-Attention sequence decoder may be faster and more expressive.

---

## 7. Important TODOs

### High Priority (Scientific)
1.  **Device-Type-Aware Loss Masks:** Ensure structurally irrelevant padding outputs do not contribute to the loss (e.g., a generator should not be penalized for getting a load parameter wrong).
2.  **Explicit Injected-Device Support:** Make direct node disturbance injections first-class entities in the device table.

### Medium Priority (Performance)
3.  **Pre-collated PyG Datasets:** Save generated PyG batches to disk as `.pt` files to bypass Polars and CPU collation during training, drastically speeding up epochs.
4.  **Target Memory Duplication:** Stop cloning identical input/target tensors if memory pressure remains critical. 

### Future Vision (Multimodal)
5.  Implement continuous wavelet transform (CWT) data loader for time-domain signals.
6.  Implement ResNet/ViT Time-Domain Encoder mapping to the shared `hidden_dim` latent space.