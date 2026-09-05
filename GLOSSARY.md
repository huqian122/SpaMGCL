# SpaMGCL Glossary

This glossary is intended for the SpaMGCL project root so Codex can keep abbreviations stable across staged implementation runs.

| Term | Full name | Meaning in this project |
|---|---|---|
| SpaMGCL | Spatial Multi-Granularity Adaptive Contrastive Learning | Planned method for spatial multi-omics integration and spatial domain clustering. |
| SMGC | Spatial multi-omics analysis framework based on granular-ball contrastive learning | Reference spatial multi-omics method and data-processing source. |
| MGCMVC | Multigranularity Information Fused Contrastive Learning With Multiview Clustering | Reference multiview clustering method and model-design source. |
| FG | Fine-grained representation | Spot-level, local, modality-specific molecular representation. |
| CG | Coarse-grained representation | Higher-level representation intended to capture tissue-domain semantics. |
| MG | Multigranularity representation | Fused representation combining FG and CG. |
| WD | Wasserstein distance | Distance used by MGCMVC-style adaptive view weighting. |
| SC | Spatial consistency | Spatial smoothness or continuity score computed over the spatial graph. |
| SNF | Spatial negative filtering | Contrastive masking rule that ignores spatial neighbors as negatives without making them positives. |
| SCR | Spatial Cluster Refinement | Optional Phase II refinement of Q using spatial neighbors before target distribution construction. |
| GCN | Graph convolutional network | Graph encoder used to obtain graph-specific views. |
| AE | Autoencoder | Encoder-decoder path used for reconstruction and fine-grained representation learning. |
| HVG | Highly variable genes | RNA feature-selection step referenced from SMGC-style preprocessing. |
| ADT | Antibody-derived tags | Protein modality in RNA-ADT spatial multi-omics datasets. |
| ATAC | Assay for transposase-accessible chromatin | Chromatin accessibility modality in RNA-ATAC datasets. |
| LSI | Latent semantic indexing | Dimensionality reduction commonly used after TF-IDF for ATAC features. |
| ARI | Adjusted Rand Index | Clustering evaluation metric. |
| NMI | Normalized Mutual Information | Clustering evaluation metric. |
| mclust | Model-based clustering | Main downstream clustering option aligned with SMGC-style reporting when available. |
