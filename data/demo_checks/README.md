# CSV accuracy checks

Upload one CSV at a time in http://localhost:3000/traffic and click Run Detection.
Each contains 100 rows, all 42 model inputs, plus label and attack_cat for scoring.
The frontend strips the two ground-truth columns before sending model inputs.

- 01_mixed_100.csv: 50 normal and 50 attack rows; use for the main accuracy demonstration.
- 02_normal_100.csv: 100 normal rows; alerts are false positives.
- 03_attacks_100.csv: 100 attack rows; normal decisions are missed attacks.

Expected counts and accuracy for the frozen v1.0.0 release are in expected_results.json.
Files are disjoint seeded random samples selected before model scoring. Exact
predictor overlap with all development partitions and duplicate predictor rows
were removed. Source: historical UNSW-NB15 official testing CSV. These small,
curated populations are not a fresh independent estimate of overall accuracy.
The normal-only set cannot measure attack recall; the attack-only set cannot
measure false-positive rate. Do not modify ground-truth labels when checking accuracy.
