# APSA

Code accompanying the manuscript **State-Aware Continual Test-Time Adaptation via
Anatomical Prior Guidance for Medical Image Segmentation**.

Anatomical Prior-guided Selection and Adaptation (APSA) is a state-aware
continual test-time adaptation framework for medical image segmentation. It
turns task-specific anatomical knowledge into spatial and structural evidence,
uses that evidence to guide online adaptation, selects a prediction for the
current input, and separately decides which model state is passed to the next
input.


![Conceptual comparison of APSA with representative test-time adaptation paradigms](assets/apsa_overview.png)


## Method

APSA has three main components:

- **Anatomical-prior-guided learning (APL)** adds task-relevant spatial or
  structural evidence to the online adaptation objective.
- **Multi-cue reliability estimation (MRE)** scores a fixed set of candidate
  predictions using the cues available for the task, including uncertainty,
  prior alignment, region size, location, and topology.
- **Risk-aware parameter reversion (RPR)** decides whether the candidate
  runtime state is retained or the task's predefined restorable variables are
  rolled back before the next input is processed.


## Experimental protocols

| Task | Source domain | Target protocol | Online unit |
|---|---|---|---|
| Cardiac MRI | M&Ms Domain A | One persistent B → C → D stream | Up to ten consecutive preprocessed 2-D slices |
| Prostate MRI | RUNMC | BMC, I2CVB, UCL, BIDMC, and HK; each target site starts from the same source state | One 3-D case |
| Fundus OD/OC | Each of the five registered domains or cohorts serves as source once | The other four form a persistent target stream | One image |

Cardiac and fundus adaptation continues across target-domain boundaries.
Prostate adaptation is continual within a site and is reinitialized for the
next site.

## Main results

The following mean DSC values are reported in the manuscript.

| Task | Source only | Reported comparator | APSA | Gain over source |
|---|---:|---:|---:|---:|
| Cardiac M&Ms | 69.72 | 79.23 (locally reproduced A3-TTA) | **80.37** | **+10.65** |
| Multi-site prostate MRI | 72.36 | 78.04 (locally evaluated PASS) | **80.37** | **+8.01** |
| Fundus OD/OC | 65.86 | 71.93 (published VPTTA result) | **74.48** | **+8.62** |


## Installation

Clone the repository and create an environment:

```bash
git clone https://github.com/lixiangcog/APSA.git
cd APSA

conda create -n apsa python=3.7.9 -y
conda activate apsa
```

The experiments used PyTorch 1.13.1 and TorchVision 0.14.1. Install the build
that matches your CUDA runtime, then install the shared dependencies:

```bash
python -m pip install -r requirements.txt
```

The three runners build on separate research codebases, so task-specific
environments may be easier to maintain:

- cardiac MRI: [A3-TTA](https://github.com/HiLab-git/A3-TTA)
- prostate MRI: [PASS](https://github.com/EndoluminalSurgicalVision-IMR/PASS)
- fundus OD/OC: [VPTTA](https://github.com/Chen-Ziyang/VPTTA)

## Data and source models



| Task | Original datasets | APSA setting |
|---|---|---|
| Cardiac MRI | [M&Ms Challenge](https://www.ub.edu/mnms/) | Domain A is the source; B, C, and D are targets |
| Prostate MRI | [NCI-ISBI 2013](https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/), [I2CVB](https://i2cvb.github.io/), and [PROMISE12](https://promise12.grand-challenge.org/Details/) | RUNMC is the source; BMC, I2CVB, UCL, BIDMC, and HK are targets |
| Fundus OD/OC | [RIM-ONE-r3](https://medimrg.webs.ull.es/), [REFUGE](https://refuge.grand-challenge.org/Details/), [ORIGA](https://doi.org/10.1109/IEMBS.2010.5626137), and [Drishti-GS1](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) | Five source-domain jobs; `REFUGE_Valid` denotes the REFUGE-Eval800 cohort |

## Run


### Cardiac M&Ms

```bash
BASE_ROOT=/abs/path/to/A3-TTA \
CHECKPOINT=/abs/path/to/source-A.pth \
OUTPUT_ROOT=/abs/path/to/outputs/cardiac \
PYTHON=/abs/path/to/python \
bash configs/run_cardiac.sh --run 0 BCD current
```


### Multi-site prostate MRI



```bash
BASE_ROOT=/abs/path/to/APSA-compatible-PASS \
CLIP_PRIOR_ROOT=/abs/path/to/MRI_prostate/clip_priors \
OUTPUT_ROOT=/abs/path/to/outputs/prostate \
GPU=0 \
PYTHON=/abs/path/to/python \
bash configs/run_prostate.sh
```

### Fundus OD/OC

```bash
BASE_ROOT=/abs/path/to/VPTTA/OPTIC \
DATASET_ROOT=/abs/path/to/Fundus \
MODEL_ROOT=/abs/path/to/source/checkpoints \
OUTPUT_ROOT=/abs/path/to/outputs/fundus \
GPU=0 \
PYTHON=/abs/path/to/python \
bash configs/run_fundus.sh
```

