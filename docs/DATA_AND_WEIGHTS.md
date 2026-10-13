# Data and weights

No medical images, annotations, derived arrays, or model weights are
distributed in this repository. Obtain every dataset from its original
provider and comply with that provider's access and use terms.

A3-TTA, PASS, and VPTTA are upstream **code dependencies**. References to their
directory layouts below describe software compatibility and do not identify
the origin or owner of a dataset.

## Cardiac M&Ms

### Original source

The cardiac experiment uses the **Multi-Centre, Multi-Vendor and Multi-Disease
Cardiac Segmentation (M&Ms) Challenge** dataset. Request the data through the
[official M&Ms access page](https://www.ub.edu/mnms/download.php); use is
subject to the organizers' [Data Use Agreement](https://www.ub.edu/mnms/documents/data_use_agreement.pdf).
Please cite [Campello et al., IEEE TMI 2021](https://doi.org/10.1109/TMI.2021.3090082).

### APSA protocol and local layout

Domain A supplies the source model and Domains B, C, and D form the persistent
target stream. Prepare legitimately obtained M&Ms data using the same
two-dimensional preprocessing and directory conventions as
[A3-TTA](https://github.com/HiLab-git/A3-TTA). Place the Source-A checkpoint
outside this repository and pass its path with `CHECKPOINT`; supply the stream
explicitly to the launcher (for example, `BCD`). A3-TTA defines the compatible
code layout, not the dataset provenance.

## Multi-site prostate MRI

### Original sources and site aliases

The six-site T2-weighted MRI benchmark is assembled from three public
collections. The site names are the domain aliases used by the experiment;
the organization follows the public
[SAML/FedDG dataset project](https://liuquande.github.io/SAML/).

| Domain | Site | Cases | APSA/PASS role | Original collection and authoritative source |
|---|---|---:|---|---|
| A | RUNMC (Radboud University Nijmegen Medical Centre) | 30 | Labeled source | [NCI-ISBI 2013 on TCIA](https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/) |
| B | BMC (Boston Medical Center) | 30 | Target stream | [NCI-ISBI 2013 on TCIA](https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/) |
| C | I2CVB / HCRUDB (Dijon-Bourgogne) | 19 | Target stream | [I2CVB project](https://i2cvb.github.io/) and [archival record](https://doi.org/10.5281/zenodo.162231) |
| D | UCL (University College London) | 13 | Target stream | [PROMISE12](https://promise12.grand-challenge.org/Details/) and [archival release](https://doi.org/10.5281/zenodo.8026660) |
| E | BIDMC (Beth Israel Deaconess Medical Center) | 12 | Target stream | [PROMISE12](https://promise12.grand-challenge.org/Details/) and [archival release](https://doi.org/10.5281/zenodo.8026660) |
| F | HK (Haukeland University Hospital, Norway) | 12 | Target stream | [PROMISE12](https://promise12.grand-challenge.org/Details/) and [archival release](https://doi.org/10.5281/zenodo.8026660) |

Here `HK` denotes Haukeland University Hospital, not the University of Hong
Kong. `I2CVB` is the collection name; the corresponding site is also reported
as HCRUDB in the source literature.

### APSA protocol and local layout

The source model is trained from the 30 labeled RUNMC cases. The complete BMC,
I2CVB, UCL, BIDMC, and HK domains (30/19/13/12/12 cases) are processed as five
independently initialized target streams. Target annotations are not used for
adaptation or state selection; they are accessed only for post-hoc evaluation.
Use the preprocessing, manifest format, and Source-RUNMC checkpoint conventions of
[PASS](https://github.com/EndoluminalSurgicalVision-IMR/PASS). PASS supplies a
convenience copy of its prepared benchmark through its own README, but it is
not the primary provenance authority for the underlying scans.

### Fixed BioMedCLIP localization maps

The reference runner consumes fixed semantic-localization maps, but the exact
maps used for the reported experiment are not redistributed. The included
`generate_biomedclip_priors.py` is an unverified legacy utility: an audit found
substantial missing and unexpected checkpoint keys with the BioMedCLIP revision
listed below. Do not use its current output as a reproduction artifact.

For an already verified experiment-generated `*_clip.npy` map, the format-only
converter creates the portable `.npz` read by the runner:

```bash
python apsa/prostate/convert_clip_priors_npy_to_npz.py \
  --root /path/to/MRI_prostate/clip_priors
```

The intended checkpoint identifier recorded by this release is
`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, revision
`9f341de24bfb00180f1b847274256e9b65a3a32e`. The packaged legacy utility
hard-codes `Prostate gland.`, but the prompt set and preprocessing provenance of
the exact reported maps are not established by this snapshot. A corrected
generator must verify checkpoint load coverage and publish that metadata plus
reference-map checksums. The converter above changes serialization only and
does not validate map provenance. Maps remain fixed during target adaptation.

## Fundus OD/OC

### Original sources and loader aliases

The identifiers in the first column are retained for loader and checkpoint
compatibility. They are not adaptation methods or dataset providers.

| Loader identifier | Dataset or official partition represented | Authoritative source |
|---|---|---|
| `RIM_ONE_r3` | RIM-ONE release 3; manifests contain 99 training and 60 evaluation images | [University of La Laguna MIAG](https://medimrg.webs.ull.es/) and [Fumero et al., CBMS 2011](https://doi.org/10.1109/CBMS.2011.5999143) |
| `REFUGE` | The 400-image REFUGE training cohort, split into 320 training and 80 evaluation entries by the experiment manifests | [Official REFUGE description](https://refuge.grand-challenge.org/Details/) and [Orlando et al., Medical Image Analysis 2020](https://doi.org/10.1016/j.media.2019.101570) |
| `ORIGA` | ORIGA-light; manifests contain 500 training and 150 evaluation images | [Zhang et al., EMBC 2010](https://doi.org/10.1109/IEMBS.2010.5626137); obtain the data from the authors or an authorized distributor |
| `REFUGE_Valid` | Compatibility alias for **REFUGE-Eval800**: the union of the official 400-image off-site validation cohort and 400-image on-site test cohort; it is not a separate dataset | [Official REFUGE description](https://refuge.grand-challenge.org/Details/) |
| `Drishti_GS` | Drishti-GS1; official 50-image training and 51-image test partitions | [Official IIIT Hyderabad page](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) and [access page](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Download.php) |

### APSA protocol and local layout

The manuscript protocol uses these five registered domains as the labeled
source in turn, with the remaining four forming that job's target stream. Use the
manifest, preprocessing, and OPTIC directory conventions of
[VPTTA](https://github.com/Chen-Ziyang/VPTTA). VPTTA defines the compatible
software layout; the table above records the underlying data provenance.

Dataset licenses and access rules apply independently. In particular, obtain
Drishti-GS1 through its official registration process. Do not commit local
images, annotations, masks, arrays, or checkpoints to this repository.
