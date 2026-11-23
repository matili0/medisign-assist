# Included inference models

The repository includes the trained SPOTER model and its metadata in
`spoter_legacy/`, plus the unchanged MediaPipe body and hand detectors in `mediapipe/`.
These files are sufficient for the desktop app, pose examples and inference tests.
The SPOTER checkpoint is the historical medical model evaluated in the README.

New training runs belong in ignored `runs/` directories. Keep these reference
weights intact when experimenting. See [attribution and model notices](../NOTICE.md).
