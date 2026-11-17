# Attribution and third-party materials

MediSign Assist was developed by **Mathis Herbreteau, Axel Bonneau and Quentin Richard**.
This repository does not assign a new open-source license to the team's original
code. Third-party notices below do not license the entire project.

## LSFB-ISOL

The three pose examples and optional prepared medical corpus are derived from LSFB-ISOL.
They retain the source dataset's **CC BY-NC-SA 4.0** terms. No videos or patient records
are bundled. The samples are real corpus-derived poses, not synthetic data.

Attribution: Meurant, Laurence (2015), *Corpus LSFB. Un corpus informatisé en libre
accès de vidéos et d'annotations de la langue des signes de Belgique francophone
(LSFB)*, Laboratoire de Langue des signes de Belgique francophone, Université de Namur.

Dataset reference: Jérôme Fink, Benoît Frenay, Laurence Meurant and Anthony Cleve
(2021), *LSFB-CONT and LSFB-ISOL: Two New Datasets for Vision-Based Sign Language
Recognition*, IJCNN. [DOI](https://doi.org/10.1109/IJCNN52387.2021.9534336).

Transformations: vocabulary selection, concatenation of body/hand coordinates,
documented exclusion of empty sequences, and a local signer-disjoint partition.
The derived poses and examples retain CC BY-NC-SA 4.0. Corpus identifiers must not
be used to attempt to identify participants.

- [Dataset website](https://lsfb.info.unamur.be/)
- [LSFB-ISOL structure](https://lsfb-team.github.io/lsfb-dataset/lsfb_isol/)
- [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## SPOTER

Matyáš Boháček and Marek Hrúz (2022), *Sign Pose-based Transformer for Word-level
Sign Language Recognition*. [Author repository](https://github.com/maty-bohacek/spoter).

The implementation adapts the architecture and preprocessing. The medical weights
are MediSign project weights, not official SPOTER weights. The existing
[Apache-2.0 notice](licenses/SPOTER-Apache-2.0.txt) is retained. The source project
provided the medical weights for noncommercial evaluation, retaining attribution
and the conditions of the training corpus; no broader permission is added here.

## MediaPipe detectors

The separate model archive contains unchanged `hand_landmarker.task` and
`pose_landmarker_lite.task` files. Copyright Google and contributors. Their source
model cards identify Apache-2.0; the included Apache notice is retained.

- [Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [Hand Tracking model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf)
- [BlazePose model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf)

Dependencies retain their own licenses and are installed separately. The original
team used AI assistance for code, tests and documentation and reviewed the resulting
implementation; the metrics originate from recorded model evaluations.
