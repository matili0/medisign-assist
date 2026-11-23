# Optional LSFB-ISOL corpus

The app and its three examples run without the full corpus. To prepare medical
training/evaluation poses from the official LSFB-ISOL server, run from the repository root:

```powershell
python src/prepare_corpus.py
python src/data.py audit --output-dir outputs/data-audit
```

For auxiliary SPOTER pretraining as well:

```powershell
python src/prepare_corpus.py --include-pretrain
python src/data.py audit --include-pretrain --output-dir outputs/full-data-audit
```

The script reads `dataset_path` from `config.json` (default `data/LSFB_ISOL`). It
fetches metadata and cleaned body/hand poses, merges them into 225-coordinate arrays,
retains raw medical arrays, and normalizes auxiliary arrays exactly once. It reuses
the recorded signer partition from the included model protocol. The corpus remains
excluded from Git; no videos or face landmarks are downloaded.

Downloads reuse validated local source files. Rerun the command after interruption.
Use `--prepare-only` to rebuild from already downloaded poses without network access.
The auxiliary subset is substantially larger; allow several GB of disk space and
time for many individual downloads. Exact counts depend on the upstream corpus.

See [the complete workflow](../docs/REPRODUCING.md),
[official dataset documentation](https://lsfb-team.github.io/lsfb-dataset/lsfb_isol/)
and [attribution and data conditions](../NOTICE.md).
