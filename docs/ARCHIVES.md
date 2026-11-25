# External checkpoint archive

The earlier seed-43 catonly_J directory and the legacy pipeline tarball are archived in the private Hugging Face dataset [silvix9/jlens-subliminal-artifacts](https://huggingface.co/datasets/silvix9/jlens-subliminal-artifacts/tree/main/archive). Access requires permission from its owner; this cleanup does not change dataset visibility.

`archive_manifest.json` records each original path, remote path, byte size, SHA-256, and the verified HF commit. These checkpoints describe an earlier attempt and are not the latest seed-43 J result used in the README figure.

Local originals are retained but ignored by Git. A current-tree removal does not remove historical Git objects: reducing ordinary clone size also requires a separately authorized history rewrite.
