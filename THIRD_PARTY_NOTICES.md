# Third-party notices

APSA's task runners are experiment overlays for established research
codebases. Users must obtain those projects, datasets, and checkpoints from
their authorized sources and comply with their terms.

| Component | Upstream project | Use in this release |
|---|---|---|
| A3-TTA | https://github.com/HiLab-git/A3-TTA | Cardiac model, loader, and adaptation substrate |
| PASS | https://github.com/EndoluminalSurgicalVision-IMR/PASS | Prostate model, loader, and online adaptation substrate |
| VPTTA | https://github.com/Chen-Ziyang/VPTTA | Fundus OPTIC model and prompt-adaptation substrate |
| BioMedCLIP | https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 | Offline prostate localization-map construction |

The inclusion of an upstream reference does not relicense that project, its
weights, or its datasets. The compact APSA repository intentionally excludes
vendored checkpoints and complete copies of the three base repositories.

Before creating a broader redistribution bundle, preserve all upstream
copyright notices and verify the license of every copied or modified file.
