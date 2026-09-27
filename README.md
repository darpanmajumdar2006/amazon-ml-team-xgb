# Amazon ML Challenge — Entity Resolution

## Current Progress

Completed:
- Initial dataset inspection
- Ground-truth match distribution analysis
- Sample-based true-match analysis
- Business-name normalization experiments
- Address normalization experiments
- Initial normalization evaluation
- Chunked full-dataset normalization pipeline

## Dataset Setup

The competition dataset is **not stored in this repository**. Obtain it using the method permitted by the competition organizers, then place the files in this layout:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
	├── test_source1.tsv
	├── test_source2.tsv
	└── test_source3.tsv
```

`eda.py` reads from the relative `dataset/` directory and writes full normalized TSV files to `normalized_data/`. Both directories are excluded from Git. Do not commit the competition data or generated full-size TSVs.

## Setup and Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run from the repository root after arranging the dataset as shown above:

```bash
python eda.py
```

The script performs EDA and sample comparisons, writes the three small CSV artifacts in the repository root, then processes the training and test source files in chunks into `normalized_data/`.

## Current Findings

The checked-in summary reflects 33 known true matched pairs:

- Raw exact name matching: **12.12%**
- Basic normalized name matching: **24.24%**
- Core-name normalization: **39.39%**
- Raw exact address matching: **3.03%**
- Basic/address-abbreviation normalized matching: **6.06%**
- Country agreement: **100%**

Overall ground-truth statistics:

- Singleton entities: **5.58%**
- Average matches per Source 1 entity: **3.46**
- Maximum matches per Source 1 entity: **11**

## Current Normalization

Business names:
- Unicode compatibility normalization
- Lowercase conversion
- `&` replaced with `and`
- Punctuation and whitespace normalization
- Common legal suffix removal for core-name comparison

Addresses:
- Basic Unicode/text normalization
- Common address-word abbreviation normalization

These results are based on a small sample and should be validated on a larger training sample before being used for a final model. See [NORMALIZATION_NOTES.md](NORMALIZATION_NOTES.md) for details.

## Next Steps

1. Validate normalization on a larger training sample.
2. Build candidate generation/blocking.
3. Generate name and address similarity features.
4. Add country and other structured features.
5. Train an entity-matching classifier or ranker.
6. Handle multiple matches per Source 1 entity.
7. Tune the match threshold using validation data.
8. Generate the final submission.