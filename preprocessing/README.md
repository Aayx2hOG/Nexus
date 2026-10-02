# UNSW-NB15 preprocessing

Run the preprocessing pipeline from the repository root:

```sh
python preprocessing/preprocess_unsw_nb15.py
```

The script preserves the official train/test split and fits every imputer,
encoder, and scaler on training rows only. It writes these model inputs to
`data/processed/`:

- `X_train_tree.npz`, `X_test_tree.npz`: sparse `float32` one-hot matrices for
  Random Forest and LightGBM.
- `X_train_ft_cat.npy`, `X_test_ft_cat.npy`: `int64` categorical token arrays
  for FT-Transformer. Zero is reserved for an unknown category.
- `X_train_ft_num.npy`, `X_test_ft_num.npy`: standardized `float32` numerical
  arrays for FT-Transformer.
- `y_train.npy`, `y_test.npy`: binary `int64` labels.
- `preprocessor.joblib`: all fitted preprocessing objects needed for inference.
- `preprocessing_metadata.json`: column order, feature names, categorical
  cardinalities/mappings, split sizes, label counts, and unseen categories.

`id` is removed because it is a row identifier. `attack_cat` is also removed
from the predictors because it directly reveals the binary `label` target.

Load the tree matrices with `scipy.sparse.load_npz`; load FT-Transformer arrays
and targets with `numpy.load`.
