# Minimal release scope

This snapshot is deliberately limited to the code needed to inspect APSA's
three task registrations and their frozen launch arguments.

## Core-file provenance

| Task | Frozen source SHA-256 | Packaged SHA-256 |
|---|---|---|
| Cardiac runner | `22959d4a7f1ede5262b8e35b2bcc719c40c2629517bd4dd14288aa6b7e43c745` | same |
| Prostate frozen runner | `179bd7675f2c2276976b7736751a1ad0e325ac01f03777a502cd4ac2ddc6501f` | `f44279c93f14022d3797b0544d22f4784715bdd495da71092ce3f1a212ed4d32` |
| Fundus runner | `ecab5915829043ced30e40a8a42b42897a9ad51a2e7f7f06cfbba3571e76883d` | same |

The core runner logic is preserved; files are renamed and placed under
`apsa/` for a clearer public layout. The packaged prostate file differs only
by removal of trailing whitespace. Launchers were path-parameterized for
portability. These hashes establish the provenance of the packaged runner
files only; they do not establish a complete end-to-end environment.

## Integration boundary

The public audit used A3-TTA `98675706ac120f08fa504632e94b3cf375b750d6`,
PASS `0b7f66d494d7353ee1b04701f943412ac5937ce7`, and VPTTA
`7e0a3776100d1350cd2bd0c9c3431522c5940027` as inspectable upstream bases.
These are audit-time compatibility points, not claimed experiment-time
revisions.

The frozen prostate runner also imports APSA-compatible changes to PASS model
construction, optimizer grouping, EMA/reset behavior, and path configuration.
Those integration changes are not present in vanilla PASS and are not included
in this compact snapshot. Consequently, `configs/run_prostate.sh` documents the
frozen invocation for a prepared compatible worktree; it is not yet a
standalone reproduction recipe.

The packaged fundus launcher is also not the manuscript Table IV recipe: it
uses a three-round stream and nonzero central/cup prior-loss weights, while the
reported task registration uses one round, prediction-derived adaptive OD
support, and no prior-supervised loss. The packaged command is retained as a
post-manuscript snapshot variant only.

The manuscript defines the cardiac online unit as ten consecutive 2-D slices.
The public loader batches up to ten entries in CSV order but does not explicitly
mark patient boundaries, so the prepared stream manifest should be checked
against that protocol before reproduction.

The packaged BioMedCLIP generator is a legacy inspection utility whose
checkpoint load coverage has not passed audit. It must not be used to recreate
the reported prostate priors until its loader is corrected and verified.

## Deliberate exclusions

- raw medical images, annotations, and patient/case identifiers;
- pretrained and adapted checkpoints;
- prediction masks and qualitative case exports;
- full logs, TensorBoard files, search/sweep outputs, and watchdog scripts;
- manuscript drafts, chat/audit notes, and private filesystem paths;
- duplicated baseline implementations and backup files;
- the APSA-compatible PASS integration patch, pending a third-party license
  audit and a fixed public revision;
- the exact one-round fundus Table IV launcher and its result manifest;
- verified prostate prior-generation code and reference-map checksums;
- a spacing-aware audit establishing physical units for cardiac and prostate
  surface-distance metrics;
- statistical scripts whose pairing or multiple-comparison family has not
  completed the final paper audit.

The previous RAGTTA-oriented repository state remains recoverable through Git
history and the `legacy-ragtta-release` tag; it is not duplicated on `main`.
