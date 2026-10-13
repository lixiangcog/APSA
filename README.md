# APSA

Reference implementation for **State-Aware Continual Test-Time Adaptation via
Anatomical Prior Guidance for Medical Image Segmentation**.

APSA follows a shared evidence–adapt–select–retain procedure while registering
task-specific anatomical evidence, mutable state, candidate predictions, and
state-reversion policies. The three experiments are therefore provided as
separate reference runners rather than presented as one identical loss or one
identical rollback rule.

## What is included

| Task | Core implementation | Reference launcher |
|---|---|---|
| Cardiac M&Ms | `apsa/cardiac/apsa_mms.py` | `configs/run_cardiac.sh` |
| Multi-site prostate MRI | `apsa/prostate/apsa_prostate.py` | `configs/run_prostate.sh` |
| Fundus OD/OC | `apsa/fundus/apsa_optic.py` | `configs/run_fundus.sh` |

The prostate folder also contains the utility used to construct fixed
BioMedCLIP localization maps. Small, label-free cardiac calibration records
are included under `configs/`.

This is intentionally a compact, code-only release. It does **not** include
medical images, annotations, predictions, model weights, full logs, sweep
outputs, or patient/case-level result files.

## Dataset provenance

The datasets come from their original benchmark providers, **not** from
A3-TTA, PASS, or VPTTA. Those repositories are software dependencies only.
No dataset is redistributed here.

| Task | Original dataset source | APSA domains |
|---|---|---|
| Cardiac MRI | [M&Ms Challenge](https://www.ub.edu/mnms/) ([access request](https://www.ub.edu/mnms/download.php)) | Source A; continual target stream B → C → D |
| Prostate MRI | [NCI-ISBI 2013](https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/) (RUNMC, BMC), [I2CVB](https://i2cvb.github.io/) (I2CVB/HCRUDB), and [PROMISE12](https://promise12.grand-challenge.org/Details/) (UCL, BIDMC, HK) | Source RUNMC; five target sites |
| Fundus OD/OC | RIM-ONE-r3, [REFUGE](https://refuge.grand-challenge.org/Details/), ORIGA, and [Drishti-GS1](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) | Five leave-one-domain-out jobs; `REFUGE_Valid` is the loader alias for REFUGE-Eval800, not a separate dataset |

The exact domain aliases, partitions, primary citations, access pages, and
code-layout mappings are documented in
[`docs/DATA_AND_WEIGHTS.md`](docs/DATA_AND_WEIGHTS.md).

## Software dependencies and environment

The task runners are thin experiment overlays and expect the corresponding
research codebase on `PYTHONPATH`. These links specify code and preprocessing
compatibility; they are **not dataset-source links**:

- cardiac: [A3-TTA](https://github.com/HiLab-git/A3-TTA);
- prostate: [PASS](https://github.com/EndoluminalSurgicalVision-IMR/PASS);
- fundus: [VPTTA](https://github.com/Chen-Ziyang/VPTTA) with its OPTIC layout.

The original experiments used Python 3.7.9, PyTorch 1.13.1, and TorchVision
0.14.1. Install PyTorch for the CUDA version available on your machine, then
install the additional packages in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the reference configurations

Each launcher requires paths through environment variables; no author-machine
path is embedded in the release.

```bash
# Cardiac, B -> C -> D, checkpointed state retention.
BASE_ROOT=/path/to/A3-TTA \
CHECKPOINT=/path/to/source-A-source-model-latest.pth \
PYTHON=python \
bash configs/run_cardiac.sh --run 0 BCD plus

# Prostate, one independently initialized run per target domain.
BASE_ROOT=/path/to/PASS \
PYTHON=python \
bash configs/run_prostate.sh

# Fundus, one source job followed by its registered target stream.
BASE_ROOT=/path/to/VPTTA/OPTIC \
DATASET_ROOT=/path/to/Fundus \
MODEL_ROOT=/path/to/source/checkpoints \
PYTHON=python \
bash configs/run_fundus.sh
```

The launchers refuse to run when required paths are missing and write only to
`outputs/` by default. Dataset layouts and checkpoint names remain those of
the cited base repositories; see `docs/DATA_AND_WEIGHTS.md`.

## Reproducibility notes

- Target labels are not consumed by the adaptation or state-selection code.
- The prostate semantic localization map is precomputed and fixed during
  online adaptation. The registered prompt is `Prostate gland.`
- The reported fundus configuration does not require a vision-language model.
- Some internal command-line names, such as `--use-apssa`, are retained for
  checkpoint and launcher compatibility; they refer to APSA.
- Evaluation units and aggregation rules must be kept identical when comparing
  methods. This repository does not mix published table values with locally
  paired statistical claims.

See `docs/RELEASE_SCOPE.md` for provenance hashes and the exact exclusions made
for this minimal public snapshot.

## Third-party code and licensing

The runners integrate with external research repositories and checkpoints,
which are governed by their respective terms. No project-wide license is
asserted for third-party components in this compact snapshot. Review
`THIRD_PARTY_NOTICES.md` before redistribution or downstream packaging.
