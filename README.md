# GILoop

GILoop is a deep learning model for detecting CTCF-mediated loops in Hi-C contact maps.

![Model architecture](./figure/architecture.png)

> **New:** GILoop now supports `.cool` input.

---

## Installation
```bash
conda create -n GIL python=3.8
conda activate GIL
pip install -r requirements.txt
```

After running the above, download `models` and `data` from [GILoop_assets](https://portland-my.sharepoint.com/:f:/g/personal/fuzhowang2-c_my_cityu_edu_hk/EpsC_y58ARNInLGjwy4yc44BNs2fKzCXNFVLUxrsrtHO2A?e=83KzE4) and **replace the corresponding files in your local directory**.

---

## Usage

GILoop supports two input formats: `.cool` files or Juicer dump text files.

### Cooler Input
```bash
python demo.py
```

API usage is self-documented in [demo.py](./demo.py). This script runs the full pipeline including data preprocessing (sequencing depth alignment, normalization, expected vector calculation, etc.), GILoop patch sampling, and model training. Note that preprocessing can be computationally intensive and may result in longer runtimes.

### Juicer Dump Text File Input
```bash
python demo_from_processed.py
```

Input files must be KR-normalized O/E matrices, and the source and target datasets should have a similar number of reads. This script skips preprocessing and runs only GILoop patch sampling and model training.