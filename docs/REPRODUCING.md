# Reproducing the workflow

Run commands from the repository root in the environment installed according to the
README. The inference model and MediaPipe detectors are included in Git. The corpus
is optional and is downloaded directly from the official LSFB server when requested.

## Verify the demo without the corpus

```powershell
python -m unittest discover -s tests -v
python src/app.py --input examples/boire.npy
```

## Download and prepare medical poses

```powershell
python src/prepare_corpus.py
python src/data.py audit --output-dir outputs/data-audit
```

The script downloads `instances.csv` and cleaned poses for the 20 medical signs:
33 body points, 21 left-hand points and 21 right-hand points. It concatenates x/y/z
coordinates into raw `(T, 225)` arrays in `data/LSFB_ISOL/landmarks/` and writes
`index.csv`. Empty source sequences are excluded by the loader. Videos are not needed.
It installs the recorded signer partition from the bundled model protocol rather
than generating a different split. See [data layout and options](../data/README.md).

Internet access and the upstream LSFB server are required. Existing source files
are validated and reused. Rerun after interruption; `--workers 2` lowers concurrency.
Use `--prepare-only` if all required poses are already downloaded. For a different
storage location, change `dataset_path` in `config.json` before downloading. Corpus
files are ignored by Git. Read [dataset terms and attribution](../NOTICE.md) first.

## Evaluate the existing model

Use a fresh output folder for every evaluation. This runs inference and writes new
results; it does not replace the recorded figures in `docs/images/`.

```powershell
python src/evaluate.py run --run models/spoter_legacy --split test --threads 1 --output-dir outputs/spoter-test
python src/evaluate.py threshold --run models/spoter_legacy --output-dir outputs/thresholds
```

Threshold selection uses validation. The upstream corpus may evolve; the saved
bundle checks the example IDs, labels and partition before evaluation. Changed
upstream inputs do not establish reproduction of the recorded scores.

## Train classical baselines

```powershell
python src/train.py classical --model all --output-dir runs/classical
python src/evaluate.py run --run runs/classical/svm --split val --threads 1 --output-dir outputs/svm-val
python src/evaluate.py run --run runs/classical/svm --split test --threads 1 --output-dir outputs/svm-test
```

This trains majority, logistic regression and SVM baselines. SVM hyperparameters are
selected on validation. Each family has its own child directory under `runs/classical`.

## Pretrain SPOTER and fine-tune medical signs

First add the auxiliary corpus: nonmedical signs with at least 50 occurrences in
the full metadata, restricted to train/validation signers. The preparation script
normalizes auxiliary poses once into `pretrain_landmarks/`. This download is much
larger than the medical subset and can take substantial time.

```powershell
python src/prepare_corpus.py --include-pretrain
python src/data.py audit --include-pretrain --output-dir outputs/full-data-audit
python src/pretrain.py --output-dir runs/pretrain
python src/train.py medical --pretrained runs/pretrain/pretrained.pt --augmentation on --output-dir runs/medical
python src/evaluate.py run --run runs/medical --split val --output-dir outputs/medical-val
python src/evaluate.py run --run runs/medical --split test --output-dir outputs/medical-test
```

To resume interrupted pretraining:

```powershell
python src/pretrain.py --resume runs/pretrain/last.pt
```

Choose a new output directory for a new experiment; completed runs are not silently
overwritten. Medical fine-tuning requires the auxiliary checkpoint and its metadata;
the supplied 20-class demo weights are not a substitute for auxiliary pretraining.
A CUDA-compatible PyTorch installation is recommended for training. The README's
lock file targets the Windows CPU demo. See each command's `--help` for options.

The exact auxiliary checkpoint behind the historical medical weights was not
recorded. These commands enable a new experiment using the included implementation;
they do not promise bit-for-bit recreation of the historical weights or scores.
