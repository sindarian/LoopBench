# LoopBench

LoopBench is a benchmarking suite for CTCF-mediated loop detection in Hi-C contact maps, comparing GILoop against two novel deep learning architectures — Enhanced UNet and LoopNet — while characterizing the effect of key parameters such as quantile thresholds, normalization strategies, patch size, and resolution on loop calling performance.

---

## Installation
```bash
conda create -n loopbench python=3.8
conda activate loopbench
pip install -r requirements.txt
```

After running the above, download `models` and `data` from [LoopBench_assets](https://portland-my.sharepoint.com/:f:/g/personal/fuzhowang2-c_my_cityu_edu_hk/EpsC_y58ARNInLGjwy4yc44BNs2fKzCXNFVLUxrsrtHO2A?e=83KzE4) and **replace the corresponding files in your local directory**.

---

## Usage

### Running Experiments

All experiments are run via `experiment_runner.py`:
```bash
python experiment_runner.py
```

The runner executes 6 experiments in sequence:

- **Experiment 1** — Exhaustive grid search over quantile thresholds (0.50–0.99) and normalization strategies (`clip`, `divide`) to identify the best performing combination.
- **Experiment 2** — Evaluates the best quantile threshold from Experiment 1 across a range of log-based normalization pipelines (e.g. `log,zscore`, `log,clip`) to identify the optimal normalization strategy.
- **Experiment 3** — Grid search over patch sizes (32, 64, 128, 224) and Hi-C resolutions (5kb, 10kb, 15kb) using the best normalization strategy from Experiment 2.
- **Experiment 4** — Reuses the Experiment 3 setup with a fixed patch size (64) and resolution (10kb) to train the base enhanced model on all chromosomes.
- **Experiment 5** — Trains a resolution-aware model by combining data from 5kb, 10kb, and 15kb resolutions into a single multi-resolution generator.
- **Experiment 6** — Compares three normalization pipelines: custom normalization (`log,zscore`), Mustache preprocessing, and a combination of both.

Before running, configure the dataset paths, cell line, and chromosomes at the top of `experiment_runner.py`:
```python
assembly = 'hg38'
bedpe_path = 'bedpe/hela.hg38.bedpe'
image_data_dir = 'data/txt_hela_100'
dataset_name = 'hela_100'
```

### Juicer Dump Text File Input
```bash
python demo_from_processed.py
```

Input files must be KR-normalized O/E matrices, and the source and target datasets should have a similar number of reads. This script skips preprocessing and runs only patch sampling and model training.