DL_Project — quick overview

- Key pieces:
  - `data_processing/` — manifest generator (`split.py`) and `MIDASDataset` (`data.py`).
  - `training_scripts/` — quick test + training entrypoints (e.g. `tets.py`, `resnet_train.py`).
  - `manifests/` — generated CSV splits (train/val/test) produced by `split.py`.
- Data: place raw files under `data/MRA-MIDAS/...` (not committed to git).
- Typical workflow:
  1. Create manifests (all the csv for the different train, val, test and unlabeled test sets) (patient-wise stratified):
	  - `python -m data_processing.split <master.xlsx> <image_root> --out_dir manifests`
  2. Quick dataset check:
	  - `python -m training_scripts.tets`
  3. Run image_cache.py to resize images before running the dataloader

- Notes:
  - Do NOT commit raw data or `__pycache__`; `.gitignore` is configured.
  - `MIDASDataset` returns dicts: `{'image','tabular','label'}`. Fit scalers/encoders on train only.

