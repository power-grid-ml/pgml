
# Implementation Details: Power Grid Multimodal State Estimation 

## Purpose

This repository implements a hierarchical machine learning architecture for power-system state estimation under partial observability, uncertain pseudo-measurements, and variable graph structure.

The target use case is power-quality-aware state estimation with:
- variable-size grids
- variable numbers of nodes and edges
- variable numbers of harmonic components
- variable numbers of attached devices per node
- explicit device-level modeling of loads, generators, voltage sources, and later direct injected disturbances

The architecture is designed to preserve device identity and harmonic structure while allowing graph-based inference over the physical network topology.

---

## Core Strategy

### Problem
Conventional models using fixed vectors of harmonic components suffer from several limitations:
- high-magnitude components dominate training loss
- fixed harmonic counts require padding or rigid input layouts
- oscilloscope-like or richer signal representations are hard to integrate
- node-level aggregation of attached devices loses physically relevant detail
- disturbance attribution to specific devices becomes difficult or impossible

### Solution
The implemented strategy is:

1. represent each simulation step as one full graph sample
2. represent node and edge measurements as variable-length token sets
3. represent attached devices explicitly rather than aggregating them
4. encode local tokens into latent vectors
5. fuse device context into node context
6. run a graph neural network over the physical topology
7. decode node, edge, and device targets
8. train with curriculum-based masking and noisy pseudo-measurements

---

## Data Representation

One `HeteroData` object corresponds to one `(dataset_id, step)`.

### Node store
Contains:
- static node features
- node measurement tokens
- node voltage targets

### Edge store
Contains:
- static edge features
- edge measurement tokens
- edge current/power targets

### Device store
Contains:
- explicit static device features
- device-to-node mapping
- parameter tokens
- spectrum tokens
- device-specific targets

This design preserves:
- multiplicity of devices per node
- device identity
- device type
- harmonic spectrum information

---

## Software Architecture

### Data pipeline
Main components:
- `topology_export.py`
- `topology.py`
- `measurement_tokenizer.py`
- `step_graph_assembler.py`
- `step_dataset.py`
- `dataloader_step.py`

### Model stack
Main components:
- `token_encoders.py`
- `masking.py`
- `fusion.py`
- `graph_state_estimator.py`
- `decoders.py`
- `multimodal_state_estimator.py`

### Training and evaluation
Main components:
- `multitask_engine.py`
- `evaluation.py`
- `main.py`

---

## Model Overview

### 1. Local encoders
Three local encoders transform token sets into fixed-size latent vectors:
- node measurement encoder
- edge measurement encoder
- device encoder

Current implementation uses:
- token-value MLP
- frequency embedding MLP
- token-type embedding
- masked mean pooling

### 2. Masking and pseudo-measurement corruption
The training objective follows a curriculum learning strategy.

#### Node and edge masking
A growing fraction of node and edge measurement latents is replaced by learnable mask tokens.

This teaches the graph model to infer the full state from only a few live measurements.

#### Device corruption
Device parameter inputs are corrupted with Gaussian noise.

Device spectra can be dropped and replaced with an unknown-spectrum token.

This simulates inaccurate pseudo-measurements and missing harmonic priors.

### 3. Node-device fusion
Explicit device latents are pooled to the node they are attached to and fused with:
- node static features
- node measurement latents
- observability indicators

### 4. Graph state estimator
A lightweight edge-aware graph neural network propagates information through the physical network.

Current implementation uses `TransformerConv`.

### 5. Decoders
The model predicts:
- node dynamic voltage tokens
- edge dynamic current/power tokens
- device parameter tokens
- device spectrum tokens

Current decoder implementation is token-conditioned using target frequency and target type.

---

## Inputs and Outputs

## Inputs

### Static inputs
- node topology features
- edge topology features
- explicit static device features
- device-to-node mappings

### Dynamic inputs
- node harmonic voltage measurements
- edge harmonic current/power measurements
- device parameters
- device spectra

### Curriculum / observability inputs
- node masking ratio
- edge masking ratio
- device noise scale
- spectrum drop probability

## Outputs

### Node outputs
- reconstructed / estimated harmonic voltage tokens

### Edge outputs
- reconstructed / estimated harmonic current or power tokens

### Device outputs
- reconstructed / estimated parameter tokens
- reconstructed / estimated spectrum tokens

---

## Training Objective

The current training engine logs:

- total loss
- encoder/decoder loss proxy
- state estimator loss proxy
- node loss
- edge loss
- device parameter loss
- device spectrum loss

At the current implementation stage:
- encoder/decoder loss and state estimator loss are still numerically identical proxies
- a stricter separation will be added later when local reconstruction and graph-estimation losses are split more explicitly

---

## Evaluation

After training, the pipeline produces:
- `training_history.json`
- `loss_curves.png`
- `validation_summary.txt`

### Validation summary
Includes:
- overall losses
- dynamic loss by target type
- frequency-wise loss summaries
- device-type-wise loss summaries

### Loss plot
Shows:
- train/validation total loss
- train/validation reconstruction loss
- train/validation state loss
- train/validation node/edge/device losses
- curriculum schedule:
  - node mask ratio
  - edge mask ratio
  - device noise scale
  - spectrum drop probability

---

## Current Simplifications

The current implementation is structurally aligned with the final architecture, but several parts are intentionally lightweight:

1. local token encoders use masked mean pooling instead of attention
2. node-device fusion uses mean pooling instead of attention
3. decoder uses token-conditioned MLP decoding instead of cross-attention or transformer decoding
4. direct injected disturbances are not yet explicit device entities
5. device loss is not yet masked by semantic validity per device type
6. static features are conditioning inputs only, not supervised targets

These simplifications are suitable for local debugging and format validation on small hardware.

---

## Important TODOs

### Highest priority
- add explicit injected-device entities for direct disturbance modeling
- add device-type-aware loss masks
- split encoder/decoder loss from graph state estimator loss more rigorously
- upgrade node-device fusion to attention-based fusion
- evaluate cross-attention or transformer-based token decoders

### Medium priority
- improve frequency embeddings
- enrich token payloads with magnitude/angle or metadata
- add richer device noise models
- add topology-aware sensor masking policies

### Lower priority
- preserve raw categorical identifiers for easier inverse mapping and reporting
- improve step indexing robustness across all data tables
- extend evaluation to static prediction tasks if those become supervised outputs

---

## Running Training

The training entry point is:

```bash
python main.py
```

This will:
- load the new step-wise dataset
- build the hierarchical graph model
- train with curriculum masking/noising
- save history, plots, and validation summaries
- optionally log metrics and artifacts to MLflow

## Scientific Motivation

The overall modeling philosophy is:

- preserve physically meaningful structure
- avoid lossy aggregation of devices
- learn under realistic partial observability
- support variable graph sizes and variable harmonic resolutions
- enable eventual disturbance attribution to individual devices

This architecture is intended as a scalable foundation for:

- dense neural state estimators
- graph neural state estimators
- later graph-transformer-based estimators
- eventual multimodal expansion with oscilloscope-derived latent features
