# climsim_test

Deep-learning parameterization of sub-grid scale processes on the [ClimSim](https://arxiv.org/abs/2306.08754) dataset.

This is the experimental codebase behind **Paraformer: Parameterization of Sub-grid Scale Processes Using Transformers** ([arXiv:2412.16763](https://arxiv.org/abs/2412.16763)). It contains the data pipeline, the baseline architectures (FCNN, MLP, CNN, RNN), the Transformer ("Paraformer") model, the hyperparameter search runs, and the ClimSim post-processing that converts raw predictions into physically-weighted MAE / RMSE / R² metrics.

All experiments here use the **v1 variable set**: 124 grid-scale inputs → 128 sub-grid scale outputs, on the low-resolution, real-geography ClimSim data.

## Motivation

Global Climate Models cannot explicitly resolve sub-grid scale physics (clouds, convection, turbulence, radiative transfer), so these processes must be approximated — *parameterized* — from grid-scale state. Existing deep-learning schemes rely on classical architectures (MLP, CNN, random forests, GANs) that treat each atmospheric state independently.

The idea tested here is that parameterization is **memory-dependent**: the sub-grid tendency at time *t* depends on the recent evolution of the column, not just its instantaneous state. Sequencing ClimSim along the time dimension and applying a Transformer encoder lets attention capture those temporal dependencies.

## Data

Low-resolution, real-geography ClimSim: 10 years, 384 unstructured spatial grid columns, 20-minute native resolution (744 GB, ~10.1M samples), subsampled for tractability.

| Split | Coverage |
| --- | --- |
| Train | years 0001-02 → 0008-01 |
| Validation | year 0008-02 → 0009-01 |

**Inputs (124):** temperature (60 levels), specific humidity (60 levels), surface pressure, insolation, surface latent heat flux, surface sensible heat flux.

**Outputs (128):** `ptend_t` (dT/dt, 60 levels), `ptend_q0001` (dq/dt, 60 levels), and 8 scalar surface variables — `NETSW`, `FLWDS`, `PRECSC`, `PRECC`, `SOLS`, `SOLL`, `SOLSD`, `SOLLD`.

### Getting the data

Download `train_input.npy`, `train_target.npy`, `val_input.npy`, `val_target.npy` from the LEAP HuggingFace mirror:

<https://huggingface.co/datasets/LEAP/subsampled_low_res/tree/main>

or run [`download_data.ipynb`](download_data.ipynb), which pulls directly via `huggingface_hub.snapshot_download`.

Post-processing additionally requires the grid-info and normalization files from the official ClimSim repo:

<https://github.com/leap-stc/ClimSim/tree/main/preprocessing/normalizations>

You need `ClimSim_low-res_grid-info.nc`, `inputs/input_{mean,min,max}.nc`, and `outputs/output_scale.nc`.

## Repository structure

| Path | Description |
| --- | --- |
| [`download_data.ipynb`](download_data.ipynb) | Fetches the subsampled ClimSim `.npy` arrays from HuggingFace. |
| [`climsim_data.ipynb`](climsim_data.ipynb) | Builds `.npy` arrays directly from the raw `mli`/`mlo` NetCDF files, for when you want a different subsampling than the published one. |
| [`data_utils.py`](data_utils.py) | ClimSim's official data utility class (variable selection, normalization, train/val splitting, metric helpers). Vendored from the upstream repo. |
| [`FCNN.ipynb`](FCNN.ipynb) | Single-hidden-layer network (124 → 64 → 128). Simplest baseline. |
| [`MLP.ipynb`](MLP.ipynb) | Deep MLP with LeakyReLU and split heads for the vertically-resolved vs. scalar outputs. Reimplementation of the strongest ClimSim baseline. |
| [`CNN.ipynb`](CNN.ipynb) | 1-D convolutional baseline. **Incomplete** — metrics are present but the architecture was not tuned. |
| [`RNN.ipynb`](RNN.ipynb) | Recurrent baseline over the same sequence construction as the Transformer, for an apples-to-apples memory comparison. |
| [`transformer_test.ipynb`](transformer_test.ipynb) | The Paraformer model: linear embedding → sinusoidal positional encoding → `nn.TransformerEncoder` → linear head. Interactive version. |
| [`transformer_test.py`](transformer_test.py) | Script version of the above, for batch submission. Includes the full post-processing block. |
| [`transformer_test.sh`](transformer_test.sh) | SLURM job script (1× V100, 450 GB RAM). |
| [`test_model.ipynb`](test_model.ipynb) | **Start here to evaluate a new architecture.** Contains a minimal network plus the complete metric pipeline — swap in your own model and everything downstream works. |
| [`ClimSim_metrics.ipynb`](ClimSim_metrics.ipynb) | Upstream ClimSim metrics/visualization notebook (Sungduk Yu), kept as the reference implementation. |
| [`reshape time space.ipynb`](reshape%20time%20space.ipynb) | Scratch notebook verifying the `(time × space, features) → (space, time, features)` reshape and sliding-window logic. Safe to delete. |
| [`metrics/`](metrics/) | Per-model MAE / RMSE / R² as CSV. `*.metrics.csv` is per output index (level-resolved); `*.metrics.lev-avg.csv` is vertically averaged. |
| [`metrics_netcdf/`](metrics_netcdf/) | Grid-wise metrics as NetCDF, for global maps and latitude–pressure cross-sections. |

## The sequence construction

Raw ClimSim arrays are flat: `(time × space, features)`. To give the Transformer temporal context, `create_sliding_window()` in [`transformer_test.py`](transformer_test.py) reshapes them per column and unfolds along time:

```
(time*space, 124)  ->  (384, time, 124)               # separate space from time
                   ->  (384, num_windows, seq, 124)   # unfold along time
                   ->  (384*num_windows, seq, 124)    # flatten into a batch
```

Predictions are made for **every** position in the window, and `get_original_shape()` reverses the transform — keeping the full first window and the last element of each subsequent window — so the output aligns 1:1 with the original time axis and the standard ClimSim metrics apply unchanged.

Note this treats the 384 columns as independent: attention runs over time only, not space. That is a deliberate simplification (ClimSim's grid is unevenly distributed, with far fewer columns near the poles) and a natural direction for future work.

## Post-processing

Raw network output is not directly comparable across variables. Every model notebook ends with the same ClimSim-standard pipeline:

1. Un-scale predictions using `output_scale.nc`.
2. Reconstruct pressure thickness `dp` from the hybrid sigma coordinate (`hyai`, `hybi`, `P0`, `state_ps`).
3. Weight vertically-resolved variables by `dp/g` (air mass per unit area).
4. Weight all variables by normalized grid-cell area.
5. Convert to a common energy unit, W/m² (`cp` for dT/dt, `lv` for dq/dt, `lv·ρ_H2O` for precipitation rates).
6. Compute MAE, RMSE, R² and write to `metrics/` and `metrics_netcdf/`.

## Results

Vertically- and horizontally-averaged MAE [W/m²] on the validation year, v1 variable set. Lower is better.

| Variable | FCNN | CNN | RNN | MLP | Transformer |
| --- | ---: | ---: | ---: | ---: | ---: |
| dT/dt (`ptend_t`) | 3.671 | 3.681 | 2.864 | 2.815 | **2.427** |
| dq/dt (`ptend_q0001`) | 5.476 | 5.418 | 4.779 | 4.631 | **4.314** |
| NETSW | 32.509 | 61.716 | 15.208 | 14.408 | **10.643** |
| FLWDS | 10.111 | 16.697 | 6.451 | 6.072 | **4.658** |
| PRECSC | 6.061 | 10.108 | 4.027 | 3.000 | **2.593** |
| PRECC | 70.895 | 71.253 | 33.967 | 39.305 | **23.176** |
| SOLS | 18.258 | 34.149 | 9.224 | 8.376 | **6.734** |
| SOLL | 22.002 | 38.000 | 11.974 | 10.737 | **9.284** |
| SOLSD | 10.297 | 13.863 | 5.766 | 4.791 | **3.952** |
| SOLLD | 8.990 | 12.212 | 5.893 | 5.004 | **4.478** |

The Transformer wins on every variable. The two sequence-aware models (RNN, Transformer) both beat the memoryless FCNN and CNN by a wide margin, supporting the core claim that temporal context matters for parameterization.

**These are the repository's own runs, not the paper's final numbers.** The published Paraformer results (Table 2 of the paper: dT/dt MAE 2.332) come from the best configuration found in the hyperparameter search — embedding dimension 256, 6 encoder layers, 4 attention heads, batch size 512, AdamW, `ReduceLROnPlateau`, 200 epochs. The checked-in metrics reflect shorter exploratory runs.

Two caveats visible in the CSVs: R² is `-inf` for `ptend_q0001` and `PRECSC` because of near-zero variance in the upper atmosphere and the near-total absence of snowfall in the tropics, respectively, and PRECC R² is large and negative for all models. This matches the paper, where those entries are suppressed.

### Hyperparameter search

`metrics/Transformer-Copy{1..13}.metrics*.csv` are the individual search runs. The dimensions explored were sequence length (2, 5, 10, 20), learning rate (1e-3, 1e-4), batch size (512, 3072), embedding dimension, encoder depth, and positional-encoding `max_len`. A window of 5–10 steps — roughly 12–24 hours of climate memory at this temporal resolution — worked best; longer windows did not help, and non-overlapping windows performed comparably to sliding ones at a fraction of the cost.

## Running

```bash
# 1. get the data
jupyter notebook download_data.ipynb

# 2. train + evaluate a baseline
jupyter notebook MLP.ipynb

# 3. train + evaluate the Transformer
jupyter notebook transformer_test.ipynb
#    or, on a cluster:
sbatch transformer_test.sh

# 4. try your own architecture
jupyter notebook test_model.ipynb
```

Use `test_model.ipynb` rather than `quickstart_example.ipynb` for evaluation — the quickstart does not compute true MAE/RMSE/R² values.

### Requirements

`torch`, `numpy`, `pandas`, `xarray`, `netCDF4`, `h5py`, `scikit-learn`, `matplotlib`, `tqdm`, `huggingface_hub`. `data_utils.py` also imports `tensorflow`, though the PyTorch path here does not use it.

The Transformer was trained on a single NVIDIA V100. The full 7-year training set is loaded into memory as dense arrays, hence the 450 GB memory request in the SLURM script — subsample further or stream the data if you have less.

### Known rough edges

- Data paths are hard-coded to `/work/sds-lab/Shuochen/climsim/`. Change them at the top of each notebook.
- `CNN.ipynb` is unfinished; its metrics should not be treated as a fair CNN baseline.
- `input.txt` is Tiny Shakespeare, left over from a GPT tutorial exercise, and unrelated to the climate work.

## Related work

The follow-up repository [`climt_paraformer`](https://github.com/shuochenw/climt_paraformer) takes the emulator from offline evaluation to **online testing** — coupling it into a column climate model and measuring whether the parameterization stays stable during prognostic integration.

## Citation

```bibtex
@article{wang2024paraformer,
  title   = {Paraformer: Parameterization of Sub-grid Scale Processes Using Transformers},
  author  = {Wang, Shuochen and Yadav, Nishant and Ganguly, Auroop R.},
  journal = {arXiv preprint arXiv:2412.16763},
  year    = {2024}
}
```

ClimSim dataset:

```bibtex
@article{yu2024climsim,
  title   = {ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation},
  author  = {Yu, Sungduk and Hannah, Walter and Peng, Liran and Lin, Jerry and Bhouri, Mohamed Aziz and Gupta, Ritwik and Lutjens, Bjorn and Will, Justus C and Behrens, Gunnar and Busecke, Julius and others},
  journal = {Advances in Neural Information Processing Systems},
  volume  = {36},
  year    = {2024}
}
```

## Acknowledgment

Supported by the U.S. Department of Defense (DoD) Strategic Environmental Research and Development Program (SERDP) (#RC20-1183) and the Indian Monsoon Mission project (#IITM/MM-III/IND-4).
