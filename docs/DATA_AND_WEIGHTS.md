# Data and weights

No data or model weights are distributed in this repository.

## Cardiac M&Ms

Prepare the two-dimensional M&Ms layout expected by A3-TTA and place the
Source-A checkpoint outside this repository. Pass its path with `CHECKPOINT`.
The reference stream is supplied explicitly to the launcher (for example,
`BCD`).

## Multi-site prostate MRI

Use the prostate layout and Source-RUNMC checkpoint expected by PASS. The five
target domains are BIDMC, BMC, HK, I2CVB, and UCL. Each domain job starts from
the registered source state; the state is inherited only within that domain's
case stream.

The fixed semantic-localization maps can be prepared with:

```bash
python apsa/prostate/generate_biomedclip_priors.py \
  --model_path /path/to/biomedclip_snapshot \
  --data_path /path/to/MRI_prostate \
  --save_path /path/to/MRI_prostate/clip_priors
```

The reference model is
`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, revision
`9f341de24bfb00180f1b847274256e9b65a3a32e`. The text prompt is
`Prostate gland.` Maps are generated offline and remain fixed during target
adaptation.

## Fundus OD/OC

Use the OPTIC directory layout expected by VPTTA. The reference jobs use
RIM-ONE-r3, REFUGE, ORIGA, REFUGE-Valid, and Drishti-GS as source domains in
turn, with the remaining domains forming each target stream.

Dataset licenses and access rules apply independently. Do not commit local
images, annotations, masks, arrays, or checkpoints to this repository.
