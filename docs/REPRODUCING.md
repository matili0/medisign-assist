# Reproducing the workflow

Run commands from the repository root after following the README installation.
The quick demo and tests require the separate `medisign-models.zip` bundle; full
corpus evaluation also needs `medisign-data.zip`. Extract either archive at the
repository root. The corpus archive provides `data/LSFB_ISOL/` with prepared medical
poses, metadata and signer groups. Archives are prepared separately from Git and
have no public download URL yet. Their checksums accompany the archives.

The raw LSFB dataset alone is not a drop-in replacement for this prepared subset.
`src/data.py` audits and loads prepared poses; it does not download or build them.
Corpus terms and transformations are described in [NOTICE.md](../NOTICE.md).

## Verify without training

```powershell
python -m unittest discover -s tests -v
python src/data.py check-env
```

## Optional evaluation of the existing model

These commands perform inference and write new results. Use a new output directory
for each run; existing recorded figures in `docs/images/` remain unchanged.

```powershell
python src/data.py audit --output-dir outputs/data-audit
python src/evaluate.py run --run models/spoter_legacy --split test --threads 1 --output-dir outputs/spoter-test
python src/evaluate.py threshold --run models/spoter_legacy --output-dir outputs/thresholds
```

Threshold analysis uses validation. Do not select a threshold on the test set.

## Optional baseline training

```powershell
python src/train.py classical --model majority --output-dir runs/classical
python src/train.py classical --model svm --output-dir runs/classical
python src/evaluate.py run --run runs/classical/svm --split val --threads 1 --output-dir outputs/svm-val
python src/evaluate.py run --run runs/classical/svm --split test --threads 1 --output-dir outputs/svm-test
```

Classical families have separate child directories. Training uses train and the
SVM hyperparameter search uses validation.

## Optional SPOTER experiments

The complete auxiliary corpus and its prepared `pretrain_landmarks/` are **not** in
the medical corpus archive. Obtain and prepare those separately before pretraining;
`--include-pretrain` audits them. The code retains the original recipe, but the
historical auxiliary checkpoint was not fully identified.

```powershell
python src/pretrain.py --help
python src/train.py medical --help
```

GPU training uses an appropriate PyTorch CUDA installation; the provided lock file
targets the Windows CPU demonstration. Scores and timings may vary across environments.
