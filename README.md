# I-SEM

I-SEM estimates direct regulatory effects from CRISPRi Perturb-seq data. It fits each knockdown's mean expression shift with a cyclic linear steady-state model:

```text
delta_c = (I - W)^-1 (B * u_c)
```

`W[i, j]` is the effect of gene j on gene i, and `B` holds one learned strength per knockdown.

This repository holds the I-SEM code and scripts for data preparation, comparison methods, simulations, scoring, biological analyses, statistics, and supplementary tables. Data and results are not bundled.

## Use I-SEM on your own dataset

```python
from src.pipeline import GRNInferencePipeline

result = GRNInferencePipeline(device="cuda:0").run(expression, interventions, gene_names)
result.adjacency  # S[i, j]: score of the edge j -> i
```

`expression` is cells x genes on the log-normalized scale, `interventions` is the per-cell target gene, and control cells are labelled `non-targeting`, `control` or `ctrl`. The defaults are the paper's model: tau 1.5, no W-dropout, 60 complementary half-splits, 120 models.

## Setup

Clone this repository with git rather than downloading a ZIP: the fit scripts read the git commit. Run the setup commands from the cloned repository root.

```bash
git clone https://github.com/ZhangLabGT/I-SEM
cd I-SEM
```

```bash
bash scripts/envs/build.sh isem inspre golem lingam igsp rice

git clone https://github.com/keleslab/DEMO_RECOMB_2026 scripts/baselines/adapre/upstream
git -C scripts/baselines/adapre/upstream switch --detach c9e3b1f

git clone https://github.com/HowardGech/RICE scripts/baselines/rice/upstream
git -C scripts/baselines/rice/upstream switch --detach 036b414
```

The build script requires `mamba` and also clones inspre, NOTEARS and GOLEM into `scripts/baselines/<method>/upstream/` at pinned commits. For I-SEM alone, install `requirements.txt` in your Python environment or run `bash scripts/envs/build.sh isem`. The simulator needs the R packages `mc2d` 0.1-22 and `mvtnorm` in `.envs/Rlib`; the `inspre` environment build installs them.

Run Python through `scripts/experiments/run_env.sh`, which uses the local environment and keeps caches inside the repository:

```bash
E="bash scripts/experiments/run_env.sh"
$E python -u <script>
```

## Demo

The demo needs no data:

```bash
bash scripts/experiments/run_env.sh python -u scripts/demo.py
```

## Data

Download the four CRISPRi screens and the reference sets below into `data/`, then run `scripts/prepare_screens.py`, which writes `data/dataset_<name>.npz`.

```bash
mkdir -p data/nadig2025
# Download the files below before preparing the screens.
$E python -u scripts/prepare_screens.py
```

| File in `data/` | Source |
|---|---|
| `k562.h5ad` | Replogle et al. 2022, [K562](https://plus.figshare.com/ndownloader/files/35773219) |
| `rpe1.h5ad` | Replogle et al. 2022, [RPE1](https://plus.figshare.com/ndownloader/files/35775606) |
| `nadig2025/GSE264667_hepg2_raw_singlecell_01.h5ad` | Nadig et al. 2025, [GEO GSE264667](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_hepg2_raw_singlecell_01.h5ad) |
| `nadig2025/GSE264667_jurkat_raw_singlecell_01.h5ad` | Nadig et al. 2025, [GEO GSE264667](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/GSE264667_jurkat_raw_singlecell_01.h5ad) |
| `9606.protein.info.v11.5.txt` (unzipped) | [STRING v11.5](https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz) |
| `9606.protein.links.detailed.v11.5.txt` (unzipped) | [STRING v11.5](https://stringdb-static.org/download/protein.links.detailed.v11.5/9606.protein.links.detailed.v11.5.txt.gz) |
| `corum_human_v5.3.txt` | [CORUM](https://mips.helmholtz-muenchen.de/corum/) release 5.3 (2026-04-14): Download page, Human complexes, format txt. Rename the file to this name. If a newer release is out, 5.3 is under Old releases. SHA-256: `5e40556eb59bb767f3396ba8c080f64c777903ffca9102345fdc35662a2e0282` |

Download CORUM in a browser; `curl` and `wget` can fail on its certificate.

The gene-name maps are included as `scripts/ensembl_to_hgnc.csv` and `scripts/ensembl_to_symbol.pkl`.

## Scripts

- `scripts/prepare_screens.py` — data preparation
- `scripts/experiments/`, `scripts/config_tests/` — I-SEM fits and scoring
- `scripts/run_all_baselines.py`, `scripts/baselines/`, `scripts/experiments/e8_adapre/` — comparison methods
- `scripts/experiments/e9_author_sims/`, `scripts/fresh_seeds/`, `scripts/sims_gpu/` — simulations
- `scripts/components/`, `scripts/denominators/` — component and exact-denominator controls
- `scripts/real_data/`, `scripts/myc/`, `scripts/adapre_path/`, `scripts/celltype_comparison/` — biology analyses
- `scripts/statistics/`, `scripts/supp_tables/`, `scripts/timing/`, `scripts/adapre_timing/` — statistics, supplement, runtime

## Baseline settings

The paper uses the following NOTEARS and GOLEM-NV settings:

| Method | Settings |
|---|---|
| NOTEARS | `lambda1=0.1`, `w_threshold=0.3`; authors' README defaults, without cross-validation |
| GOLEM-NV | Initialize from GOLEM-EV with `lambda_1=2e-2`, then fit GOLEM-NV with `lambda_1=2e-3`; `lambda_2=5` for both; postprocess with `graph_thres=0.3`. Simulations use CPU with seed 1. |

For these observational baselines, real-data inputs pool all cells, standardize genes against controls, and use a seeded 10,000-cell subsample. Simulation inputs are supplied as generated.

The inspre simulation settings are in the authors' [Rmd directory at commit 7016697](https://github.com/brielin/inspre/tree/7016697/Rmd).

