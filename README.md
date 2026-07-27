# HDGN-PDG

**Heterophily-Aware Dual-Branch Graph Neural Network for Pan-Cancer Driver Gene Prioritisation**

<p align="center">
  <img src="figures/architecture.png" alt="HDGN-PDG Architecture" width="800"/>
</p>

<p align="center">
  <b>Figure 1.</b> HDGN-PDG architecture.
  <b>(a)</b> Multi-omics features (N × 64) and PPI network topology are processed by two parallel branches.
  <b>(b)</b> The ACM branch encodes four omics modalities via modality-specific projections, then applies Adaptive Channel Mixing convolution with low-pass, high-pass, and identity channels. The topology branch maps 7 z-normalised structural features through a two-layer MLP.
  <b>(c)</b> A per-node GatedFusion module produces a sigmoid-gated interpolation: <i>g ⊙ h<sub>acm</sub> + (1 − g) ⊙ h<sub>topo</sub></i>.
  <b>(d)</b> A classification head outputs per-gene driver probability.
</p>

---

## Abstract

Accurate identification of cancer driver genes is fundamental to understanding tumorigenesis and advancing precision oncology. Graph neural network (GNN) methods that integrate multi-omics profiles with protein–protein interaction (PPI) networks have substantially improved driver gene prediction; however, existing approaches suffer from three persistent limitations: oversmoothing of driver-specific signals in heterophilic PPI neighbourhoods where driver genes are outnumbered by functionally dissimilar non-driver neighbours, loss of omics-type identity caused by flat multi-omics concatenation prior to graph operations, and the use of globally fixed fusion weights that cannot accommodate gene-level variation in the relative informativeness of molecular versus topological evidence.

We present **HDGN-PDG**, a heterophily-aware dual-branch graph neural network for pan-cancer driver gene prioritisation. The model encodes four TCGA omics modalities through modality-specific projection blocks, applies Adaptive Channel Mixing (ACM) graph convolution with low-pass, high-pass, and identity channels to handle heterophilic aggregation, and learns structural gene positions via a parallel topology MLP branch. A per-node GatedFusion module produces a sigmoid-gated, gene-specific interpolation of both branches.

Under 10 repetitions of 5-fold cross-validation across six PPI networks, HDGN-PDG achieves the highest AUPRC on five of six networks and the highest AUROC on all six, outperforming state-of-the-art methods including SGCD, HGDC, and EMOGI. Ablation experiments confirm that the OmicsEncoder and topology branch individually contribute the largest performance gains. A consensus pipeline across six PPI networks yields 326 predicted driver genes with 93.3% independent support from CancerMine and CCGD, enriched in actionable hallmark processes including receptor tyrosine kinase signalling, focal adhesion, and tumour angiogenesis. A three-stage novelty filter surfaces 51 high-confidence candidates absent from the Cancer Gene Census and NCG 7.2, providing a prioritised resource for functional screening and clinical association studies aimed at expanding the repertoire of actionable cancer targets.

---

## Repository Structure

```
HDGN-PDG/
├── hdgn_pdg.py              # Full model: training, evaluation, and cross-validation
├── requirements.txt          # Pinned dependencies (exact versions used in the paper)
├── figures/
│   └── architecture.png      # Architecture diagram (Figure 1)
├── data/
│   └── README.md             # Instructions for obtaining the dataset
├── LICENSE
└── README.md
```

## Requirements

- Python 3.11+
- CUDA 12.8 (for GPU training)

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` pins the exact versions used to produce results in the manuscript:

```
torch==2.11.0
torch-geometric==2.9.0
scikit-learn==1.6.1
numpy==2.0.2
scipy==1.16.3
```

## Usage

### Cross-validation (reproduce paper results)

```bash
python hdgn_pdg.py --data data/dataset_MULTINET_ten_5CV.pkl --reps 10 --seed 42
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | `dataset_MULTINET_ten_5CV.pkl` | Path to dataset pickle file |
| `--reps` | `2` | Number of CV repetitions (use `10` to reproduce paper results) |
| `--seed` | `42` | Global random seed for full reproducibility |
| `--hidden_dim` | `128` | Hidden dimension for ACM layers and TopoMLP |
| `--n_layers` | `2` | Number of stacked ACM convolution layers |
| `--dropout` | `0.4` | Dropout rate |
| `--temperature` | `3.0` | ACM softmax temperature |
| `--lr` | `1e-3` | Adam learning rate |
| `--epochs` | `200` | Maximum training epochs per fold |
| `--patience` | `30` | Early stopping patience (epochs) |

Run `python hdgn_pdg.py --help` for the full list.

### Expected output

The script prints per-fold AUPRC/AUROC, ACM channel mixing weights (LP/HP/ID), gate mean values, and a topology feature importance table aggregated across all folds:

```
CROSS-VALIDATION SUMMARY
============================================================
  AUPRC   : 0.8135 ± 0.0165
  AUROC   : 0.8842 ± 0.0112
============================================================
```

## Reproducibility

Deterministic execution is enforced via `seed_everything()`, which sets `CUBLAS_WORKSPACE_CONFIG`, seeds Python/NumPy/PyTorch RNGs, enables `torch.use_deterministic_algorithms(True)`, and disables cuDNN auto-tuning. Results are bitwise reproducible on the same hardware and software configuration.

## Data

The dataset pickle file contains multi-omics features (mutation frequency, copy number alteration, methylation, and gene expression across 16 TCGA cancer types), PPI network edge indices, gene labels, and pre-defined cross-validation splits. See `data/README.md` for download instructions and preprocessing details.

## Citation

```bibtex
@article{hdgn_pdg_2026,
  title   = {HDGN-PDG: A Heterophily-Aware Dual-Branch Graph Neural Network
             for Pan-Cancer Driver Gene Prioritisation},
  author  = {},
  journal = {},
  year    = {2026},
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
