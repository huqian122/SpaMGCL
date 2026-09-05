# Dataset File Map

Status: Phase 0 path audit artifact, 2026-09-05

## Roots

| Environment | Root | Status |
|---|---|---|
| Local Mac | `/Users/qianhu/博士论文/多角度聚类医学论文/SpaMGCL/Expert Systems SMGC/SMGC-data/` | exists |
| Colab | `/content/drive/MyDrive/SMGC-data/` | configured destination; not locally inspectable |

The implementation must resolve relative dataset paths against `data.root`.
Neither root may be hard-coded in model or experiment code.

## Registry

| Short name | Relative directory | RNA | ADT/ATAC | Expected modalities |
|---|---|---|---|---|
| Simulation | `simulation/` | `adata_RNA.h5ad` | `adata_ADT.h5ad` | RNA + ADT |
| HLN-A1 | `Human_Lymph_Nodes/A1/` | `adata_RNA.h5ad` | `adata_ADT.h5ad` | RNA + ADT |
| HLN-D1 | `Human_Lymph_Nodes/D1/` | `adata_RNA.h5ad` | `adata_ADT.h5ad` | RNA + ADT |
| E18.5 | `E18.5_mouse_brain/` | `adata_RNA.h5ad` | `adata_ATAC.h5ad` | RNA + ATAC |
| S2-E15 | `Mouse_Embryos_S2/E15/` | `adata_RNA.h5ad` | `adata_ATAC.h5ad` | RNA + ATAC |
| S2-E18 | `Mouse_Embryos_S2/E18/` | `adata_RNA.h5ad` | `adata_ATAC.h5ad` | RNA + ATAC |

## Existence and file-size audit

All 12 expected local `.h5ad` paths exist as of 2026-09-05. Sizes were checked
with filesystem metadata only; no complete matrix was read.

| Dataset | RNA bytes | ADT/ATAC bytes |
|---|---:|---:|
| Simulation | 16,766,208 | 4,293,136 |
| HLN-A1 | 56,568,662 | 1,144,564 |
| HLN-D1 | 31,603,009 | 1,107,679 |
| E18.5 | 280,626,648 | 222,785,147 |
| S2-E15 | 56,407,141 | 106,370,269 |
| S2-E18 | 53,621,296 | 105,533,734 |

## Expected dataset facts from the specification

The following are planning expectations and still require verification from
the actual AnnData objects:

| Dataset | Expected spots | Expected clusters |
|---|---:|---:|
| Simulation | 1296 | 5 |
| HLN-A1 | 3484 | 10 |
| HLN-D1 | 3359 | 11 |
| E18.5 | 2129 | 14 |
| S2-E15 | 1939 | 15 |
| S2-E18 | 2248 | 16 |

## Required Phase 1 inspection

For every file, inspect bounded metadata and summaries for:

- `adata.shape`
- `adata.X` type, sparsity, dtype, and value range summary
- `obs.columns`, `var.columns`
- `obsm.keys()`, `uns.keys()`, `layers.keys()`
- `adata.raw`
- spot identifiers and cross-modality order
- candidate coordinate and label fields
- raw/normalized/PCA/LSI processing state

Do not assume `obsm['spatial']`, `obs['label']`, raw counts, or any other field
before this inspection.

## Path conflict rule

The supplied path note includes a historical example path with a different
project-directory spelling. The actual local filesystem path above is the
authoritative path for this audit.

