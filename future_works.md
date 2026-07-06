# Future Work & Architectural Roadmap

This document outlines the planned improvements, architectural refinements, and engineering scaling targets for the **Protein-FM-HiREF** framework. Contributions, issues, or discussions regarding these roadmap items are highly welcome.

## Data Engineering & Distributed Scaling
* **Metadata-Driven DataLoader:** Preprocess massive structural datasets by compiling deep metadata index files for all structural entries. This will minimize disk I/O bottlenecks and localized runtime dataset parsing overhead during training.
* **Distributed Data Parallel (DDP) Support 🌟:** Refactor the core training loop to natively operate under PyTorch DDP conditions, enabling seamless scale-out across multi-GPU and multi-node clusters.
* **Dynamic Masking in HiRef:** Extend the `Hierarchical Optimal Refinement (HiRef)` global dataset alignment function to natively handle structural and padding masks, ensuring seamless compatibility with variable-length protein sequences.
* **Optimizer Re-configuration:** Re-assess and dynamically tune learning rate schedules and configurations in `configure_optimizers` specifically optimized for massive, high-throughput structural biology data regimes.

## Deep Learning Architecture & Representation Learning
* **Cross-Representation Communication:** Establish a dedicated bidirectional attention or cross-talk mechanism between the Single (`s`) and Pair (`p`) representation streams to maximize structural context sharing.
* **Add self-conditioning 🌟:** Self explannatory 
* **Conditioning Placeholders Expansion:** Scalably expand current placeholders in the single-feature network to accommodate future discrete and continuous conditional variables:
  * **Amino Acid Sequences** (for conditional generation and co-design)
  * **Latent Side-Chain Vectorizations** (enabling full all-atom structural generation)
* **Attention Registers 🌟:** Experiment with vision-transformer-style "attention registers" within the iterative structure module to isolate high-norm outliers and prevent structural attention-sink anomalies. (La-Proteina inspired)
* **Frenet-Serret Boundary Frame Interpolation 🌟:** Optimize the geometric parameterization at the sequence boundary. Currently, the final residue frame is duplicated to handle the N- and C-terminus. Implement and experiment with an extrapolation strategy that projects the final residue outward to construct a true, distinct boundary frame, mitigating edge-effect artifacts during generation.

## Sampling Trajectories & Generative Mechanics
* **Alternative Time-Step Sampling:** Move beyond standard uniform distribution time-step sampling ($t$). Explore alternative probability paths, such as `mix_unif_beta` (mixed uniform and Beta distributions), to bias training trajectories toward critical boundaries in the vector field.
* **In Silico Data Curriculum:** Incorporate lower-quality *in silico* predicted data into the training pipeline strictly at low time-steps ($t$). Gate this data dynamically based on the model's confidence and predicted accuracy metrics.
* **Inference-Time Scaling 🌟:** Develop search-based runtime trajectory scaling or reinforcement-style sampling rollouts to systematically optimize generated backbones during the sampling phase.

## Cross-Domain Verification & Benchmarking
* **Domain-Agnostic Generative Verification 🌟:** Abstract and port the core optimal transport Flow Matching framework to standard image domains (e.g., MNIST, ImageNet). Evaluate framework performance using traditional computer vision benchmarks like Fréchet Inception Distance (FID) to validate underlying architectural efficiency independent of structural biology constraints.