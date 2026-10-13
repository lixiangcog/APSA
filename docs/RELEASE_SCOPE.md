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
portability.

## Deliberate exclusions

- raw medical images, annotations, and patient/case identifiers;
- pretrained and adapted checkpoints;
- prediction masks and qualitative case exports;
- full logs, TensorBoard files, search/sweep outputs, and watchdog scripts;
- manuscript drafts, chat/audit notes, and private filesystem paths;
- duplicated baseline implementations and backup files;
- statistical scripts whose pairing or multiple-comparison family has not
  completed the final paper audit.

The previous RAGTTA-oriented repository state remains recoverable through Git
history and the `legacy-ragtta-release` tag; it is not duplicated on `main`.
