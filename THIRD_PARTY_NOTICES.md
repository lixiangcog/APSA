# Third-party notices

APSA's task runners are experiment overlays for established research
codebases. The projects listed below are software dependencies, not the
original dataset providers. Users must obtain each codebase, dataset, and
checkpoint from its respective authorized source and comply with its terms;
dataset provenance is listed separately in
[`docs/DATA_AND_WEIGHTS.md`](docs/DATA_AND_WEIGHTS.md).

| Component | Upstream project | Audit-time revision | Use in this release |
|---|---|---|---|
| A3-TTA | https://github.com/HiLab-git/A3-TTA | `98675706ac120f08fa504632e94b3cf375b750d6` | Cardiac model, loader, and adaptation substrate; no license file was found at this revision |
| PASS | https://github.com/EndoluminalSurgicalVision-IMR/PASS | `0b7f66d494d7353ee1b04701f943412ac5937ce7` | Prostate provenance base; vanilla PASS lacks the APSA integration and no license file was found at this revision |
| VPTTA | https://github.com/Chen-Ziyang/VPTTA | `7e0a3776100d1350cd2bd0c9c3431522c5940027` | Fundus OPTIC model and prompt-adaptation substrate; MIT notice included below |
| BioMedCLIP | https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 | `9f341de24bfb00180f1b847274256e9b65a3a32e` | Intended model for offline prostate-map construction; the packaged legacy loader is not verified |

The revisions above identify the bases used for the current public-code audit;
they are not represented as recovered experiment-time commits. See the root
README and `docs/RELEASE_SCOPE.md` for the current integration boundary.

The VPTTA MIT license and upstream copyright notice are reproduced in
[`THIRD_PARTY_LICENSES/VPTTA_LICENSE`](THIRD_PARTY_LICENSES/VPTTA_LICENSE).
No license file was present in the audited A3-TTA or PASS revisions. Their
absence does not imply permission to redistribute derived files; obtain
clarification from the respective authors before broader reuse.

The inclusion of an upstream reference does not relicense that project, its
weights, or its datasets. The compact APSA repository intentionally excludes
vendored checkpoints and complete copies of the three base repositories.

Before creating a broader redistribution bundle, preserve all upstream
copyright notices and verify the license of every copied or modified file.
