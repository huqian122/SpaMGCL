# DATA PROFILE

Status: Implementation Phase 1 data inspection
Date: 2026-09-05

## Inspection boundary

- Files were opened with AnnData `backed='r'`.
- Complete `adata.X` values were not read or materialized.
- Exact X min/max values are therefore `UNRESOLVED_NOT_READ`.
- No preprocessing, graph construction, model code, or training ran.
- Config: `/Users/qianhu/博士论文/多角度聚类医学论文/SpaMGCL/SpaMGCL/configs/phase1_inspection.yaml`
- Resolved `data.root`: `/Users/qianhu/博士论文/多角度聚类医学论文/SpaMGCL/Expert Systems SMGC/SMGC-data`

## File-level results

### RNA: `/Users/qianhu/博士论文/多角度聚类医学论文/SpaMGCL/Expert Systems SMGC/SMGC-data/simulation/adata_RNA.h5ad`
- Status: `OK`
- `adata.shape`: `(1296, 800)`
- `adata.X.type`: `h5py._hl.dataset.Dataset`
- `adata.X.dtype`: `float64`
- `adata.X.sparsity`: `dense_or_backed`
- `adata.X.min`: `UNRESOLVED_NOT_READ`
- `adata.X.max`: `UNRESOLVED_NOT_READ`
- `adata.obs.columns`: `ground_truth`
- `adata.var.columns`: `[]`
- `adata.obsm.keys()`: `nsfac`, `spatial`, `spfac`
- `adata.uns.keys()`: `log1p`
- `adata.layers.keys()`: `counts`
- `adata.raw`: `False`
- Candidate label fields: `ground_truth`
- Candidate spatial fields: `spatial`
- Spot ID count: `1296`

### ADT: `/Users/qianhu/博士论文/多角度聚类医学论文/SpaMGCL/Expert Systems SMGC/SMGC-data/simulation/adata_ADT.h5ad`
- Status: `OK`
- `adata.shape`: `(1296, 200)`
- `adata.X.type`: `h5py._hl.dataset.Dataset`
- `adata.X.dtype`: `float64`
- `adata.X.sparsity`: `dense_or_backed`
- `adata.X.min`: `UNRESOLVED_NOT_READ`
- `adata.X.max`: `UNRESOLVED_NOT_READ`
- `adata.obs.columns`: `ground_truth`
- `adata.var.columns`: `[]`
- `adata.obsm.keys()`: `nsfac`, `spatial`, `spfac`
- `adata.uns.keys()`: `log1p`
- `adata.layers.keys()`: `counts`
- `adata.raw`: `False`
- Candidate label fields: `ground_truth`
- Candidate spatial fields: `spatial`
- Spot ID count: `1296`

### RNA: `/Users/qianhu/博士论文/多角度聚类医学论文/SpaMGCL/Expert Systems SMGC/SMGC-data/Human_Lymph_Nodes/A1/adata_RNA.h5ad`
- Status: `OK`
- `adata.shape`: `(3484, 18085)`
- `adata.X.type`: `anndata._core.sparse_dataset._CSRDataset`
- `adata.X.dtype`: `float32`
- `adata.X.sparsity`: `dense_or_backed`
- `adata.X.min`: `UNRESOLVED_NOT_READ`
- `adata.X.max`: `UNRESOLVED_NOT_READ`
- `adata.obs.columns`: `Spatial_Label`
- `adata.var.columns`: `gene_ids`, `feature_types`, `genome`
- `adata.obsm.keys()`: `spatial`
- `adata.uns.keys()`: `[]`
- `adata.layers.keys()`: `[]`
- `adata.raw`: `False`
- Candidate label fields: `Spatial_Label`
- Candidate spatial fields: `spatial`
- Spot ID count: `3484`

### ADT: `/Users/qianhu/博士论文/多角度聚类医学论文/SpaMGCL/Expert Systems SMGC/SMGC-data/Human_Lymph_Nodes/A1/adata_ADT.h5ad`
- Status: `OK`
- `adata.shape`: `(3484, 31)`
- `adata.X.type`: `anndata._core.sparse_dataset._CSRDataset`
- `adata.X.dtype`: `float32`
- `adata.X.sparsity`: `dense_or_backed`
- `adata.X.min`: `UNRESOLVED_NOT_READ`
- `adata.X.max`: `UNRESOLVED_NOT_READ`
- `adata.obs.columns`: `Spatial_Label`
- `adata.var.columns`: `gene_ids`, `feature_types`, `genome`
- `adata.obsm.keys()`: `spatial`
- `adata.uns.keys()`: `[]`
- `adata.layers.keys()`: `[]`
- `adata.raw`: `False`
- Candidate label fields: `Spatial_Label`
- Candidate spatial fields: `spatial`
- Spot ID count: `3484`

### RNA: `/Users/qianhu/博士论文/多角度聚类医学论文/SpaMGCL/Expert Systems SMGC/SMGC-data/E18.5_mouse_brain/adata_RNA.h5ad`
- Status: `OK`
- `adata.shape`: `(2129, 32285)`
- `adata.X.type`: `h5py._hl.dataset.Dataset`
- `adata.X.dtype`: `float32`
- `adata.X.sparsity`: `dense_or_backed`
- `adata.X.min`: `UNRESOLVED_NOT_READ`
- `adata.X.max`: `UNRESOLVED_NOT_READ`
- `adata.obs.columns`: `Sample`, `TSSEnrichment`, `ReadsInTSS`, `ReadsInPromoter`, `ReadsInBlacklist`, `PromoterRatio`, `PassQC`, `NucleosomeRatio`, `nMultiFrags`, `nMonoFrags`, `nFrags`, `nDiFrags`, `Gex_RiboRatio`, `Gex_nUMI`, `Gex_nGenes`, `Gex_MitoRatio`, `BlacklistRatio`, `array_col`, `array_row`, `ReadsInPeaks`, `FRIP`, `ATAC_Clusters`, `RNA_Clusters`, `Combined_Clusters`, `Combined_Clusters_annotation`, `src`
- `adata.var.columns`: `type`, `name`, `interval`
- `adata.obsm.keys()`: `[]`
- `adata.uns.keys()`: `[]`
- `adata.layers.keys()`: `[]`
- `adata.raw`: `False`
- Candidate label fields: `ATAC_Clusters`, `Combined_Clusters`, `Combined_Clusters_annotation`, `RNA_Clusters`
- Candidate spatial fields: `[]`
- Spot ID count: `2129`

### ATAC: `/Users/qianhu/博士论文/多角度聚类医学论文/SpaMGCL/Expert Systems SMGC/SMGC-data/E18.5_mouse_brain/adata_ATAC.h5ad`
- Status: `OK`
- `adata.shape`: `(2129, 161461)`
- `adata.X.type`: `anndata._core.sparse_dataset._CSRDataset`
- `adata.X.dtype`: `int64`
- `adata.X.sparsity`: `dense_or_backed`
- `adata.X.min`: `UNRESOLVED_NOT_READ`
- `adata.X.max`: `UNRESOLVED_NOT_READ`
- `adata.obs.columns`: `Sample`, `TSSEnrichment`, `ReadsInTSS`, `ReadsInPromoter`, `ReadsInBlacklist`, `PromoterRatio`, `PassQC`, `NucleosomeRatio`, `nMultiFrags`, `nMonoFrags`, `nFrags`, `nDiFrags`, `Gex_RiboRatio`, `Gex_nUMI`, `Gex_nGenes`, `Gex_MitoRatio`, `BlacklistRatio`, `array_col`, `array_row`, `ReadsInPeaks`, `FRIP`, `ATAC_Clusters`, `RNA_Clusters`, `Combined_Clusters`, `Combined_Clusters_annotation`, `src`
- `adata.var.columns`: `[]`
- `adata.obsm.keys()`: `[]`
- `adata.uns.keys()`: `[]`
- `adata.layers.keys()`: `[]`
- `adata.raw`: `False`
- Candidate label fields: `ATAC_Clusters`, `Combined_Clusters`, `Combined_Clusters_annotation`, `RNA_Clusters`
- Candidate spatial fields: `[]`
- Spot ID count: `2129`

## Spot ID alignment

### Simulation
- Status: `MATCH`
- Reference modality: `RNA`
- Spot count: `1296`
- Compared `obs_names` in their stored order.

### HLN-A1
- Status: `MATCH`
- Reference modality: `RNA`
- Spot count: `3484`
- Compared `obs_names` in their stored order.

### E18.5
- Status: `MATCH`
- Reference modality: `RNA`
- Spot count: `2129`
- Compared `obs_names` in their stored order.

## Field decisions

Candidate fields are reported by name pattern only. A final label or spatial field is not selected in Phase 1 unless its semantics are verified from the dataset metadata.

- Label field decision: `UNRESOLVED` pending semantic verification; known aliases such as `ground_truth` and `Spatial_Label` are listed only as candidates.
- Spatial field decision: `UNRESOLVED` pending shape/content verification.
- Preprocessing state: `UNRESOLVED` until bounded value diagnostics are authorized and completed.
