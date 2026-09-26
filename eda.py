from pathlib import Path
for f in Path("dataset").rglob("*.tsv"):
    size_mb = f.stat().st_size / (1024**2)

    with open(f, "rb") as file:
        rows = sum(1 for _ in file) - 1

    print(f"{f.name:30} {rows:>12,} rows   {size_mb:>10.1f} MB")

import pandas as pd
s1 = pd.read_csv(
    r"C:\Users\DARPAN\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\test\test_source1.tsv",
    sep="\t",
    nrows=100_000
)
s2 = pd.read_csv(
    r"C:\Users\DARPAN\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\test\test_source2.tsv",
    sep="\t",
    nrows=100_000
)
s3 = pd.read_csv(
    r"C:\Users\DARPAN\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\test\test_source3.tsv",
    sep="\t",
    nrows=100_000
)
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:

    print("\n==========", name, "==========")

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing:")
    print(df.isna().sum())

    print("\nCountries:")
    print(df["country"].value_counts(dropna=False))

    print("\nExamples:")
    print(
        df[
            ["business_name", "business_address", "country"]
        ].sample(10, random_state=42)
    )
gt = pd.read_csv(
    r"C:\Users\DARPAN\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv",
    sep="\t",
    
)

print("Ground truth shape:", gt.shape)
print("\nColumns:")
print(gt.columns.tolist())
print("\nFirst 10 rows:")
print(gt.head(10))
gt["num_matches"] = (
    gt["matched_entity_ids"]
    .fillna("")
    .apply(
        lambda x: 0 if not str(x).strip()
        else len(str(x).split(","))
    )
)

print("\nMatch distribution:")
print(gt["num_matches"].value_counts().sort_index())

print("\nSingleton percentage:",
      (gt["num_matches"] == 0).mean() * 100)

print("\nAverage matches per S1:",
      gt["num_matches"].mean())

print("\nMaximum matches for one S1:",
      gt["num_matches"].max())
print("\nMatch distribution:")
print(gt["num_matches"].value_counts().sort_index())

print("\nSingleton percentage:",
      (gt["num_matches"] == 0).mean() * 100)

print("\nAverage matches per S1:",
      gt["num_matches"].mean())

print("\nMaximum matches for one S1:",
      gt["num_matches"].max())
sample_gt = gt[gt["num_matches"] > 0].sample(
    10,
    random_state=42
)
print(
    sample_gt[
        ["source1_entity_id", "matched_entity_ids", "num_matches"]
    ].to_string(index=False)
)
sample_s1_ids = set(sample_gt["source1_entity_id"])


def concat_if_present(frames, label):
    if not frames:
        print(f"\nNo matching rows found for {label}.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


train_s1_matches = []

for chunk in pd.read_csv(
    r"C:\Users\DARPAN\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source1.tsv",
    sep="\t",
    chunksize=100_000
):
    matched = chunk[
        chunk["entity_id"].isin(sample_s1_ids)
    ]

    if not matched.empty:
        train_s1_matches.append(matched)

train_s1_matches = concat_if_present(train_s1_matches, "train source 1")
print(train_s1_matches)
matched_ids = set()

for value in sample_gt["matched_entity_ids"]:
    if pd.notna(value) and str(value).strip():
        matched_ids.update(
            str(value).split(",")
        )

print("Number of matched records:", len(matched_ids))
print("Example IDs:", list(matched_ids)[:10])
train_s2_matches = []

for chunk in pd.read_csv(
    r"C:\Users\DARPAN\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source2.tsv",
    sep="\t",
    chunksize=100_000
):
    matched = chunk[
        chunk["entity_id"].isin(matched_ids)
    ]

    if not matched.empty:
        train_s2_matches.append(matched)

train_s2_matches = concat_if_present(train_s2_matches, "train source 2")
train_s3_matches = []

for chunk in pd.read_csv(
    r"C:\Users\DARPAN\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source3.tsv",
    sep="\t",
    chunksize=100_000
):
    matched = chunk[
        chunk["entity_id"].isin(matched_ids)
    ]

    if not matched.empty:
        train_s3_matches.append(matched)

train_s3_matches = concat_if_present(train_s3_matches, "train source 3")
print("SOURCE 1")
print(
    train_s1_matches[
        ["entity_id", "business_name", "business_address", "country"]
    ].to_string(index=False)
)
print("SOURCE 2")
print(train_s2_matches[
        ["entity_id", "business_name", "business_address", "country"]
    ].to_string(index=False)
)
print("SOURCE 3")
print(
    train_s3_matches[
        ["entity_id", "business_name", "business_address", "country"]
    ].to_string(index=False)
)

import pandas as pd


# 1. Select 10 random S1 entities that have at least one match

sample_gt = gt[gt["num_matches"] > 0].sample(
    10,
    random_state=42
)

print("\nSelected S1 entities:")
print(
    sample_gt[
        ["source1_entity_id", "matched_entity_ids", "num_matches"]
    ].to_string(index=False)
)



# 2. Store the selected S1 IDs


sample_s1_ids = set(
    sample_gt["source1_entity_id"]
)



# 3. Find those S1 records in the training Source 1 file

train_s1_matches = []

for chunk in pd.read_csv(
    r"C:\Users\DARPAN\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source1.tsv",
    sep="\t",
    chunksize=100_000
):

    matched = chunk[
        chunk["entity_id"].isin(sample_s1_ids)
    ]

    if not matched.empty:
        train_s1_matches.append(matched)


# Safely combine the chunks
if train_s1_matches:
    train_s1_matches = pd.concat(
        train_s1_matches,
        ignore_index=True
    )
else:
    train_s1_matches = pd.DataFrame()



# 4. Extract ALL S2/S3 IDs that are true matches

matched_ids = set()

for value in sample_gt["matched_entity_ids"]:

    if pd.notna(value) and str(value).strip():

        ids = str(value).split(",")

        matched_ids.update(ids)


print(
    "\nNumber of matched S2/S3 records:",
    len(matched_ids)
)

print(
    "Example matched IDs:",
    list(matched_ids)[:10]
)



# 5. Find the matching records in Source 2


train_s2_matches = []

for chunk in pd.read_csv(
    r"C:\Users\DARPAN\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source2.tsv",
    sep="\t",
    chunksize=100_000
):

    matched = chunk[
        chunk["entity_id"].isin(matched_ids)
    ]

    if not matched.empty:
        train_s2_matches.append(matched)


if train_s2_matches:
    train_s2_matches = pd.concat(
        train_s2_matches,
        ignore_index=True
    )
else:
    train_s2_matches = pd.DataFrame()



# 6. Find the matching records in Source 3

train_s3_matches = []

for chunk in pd.read_csv(
    r"C:\Users\DARPAN\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source3.tsv",
    sep="\t",
    chunksize=100_000
):

    matched = chunk[
        chunk["entity_id"].isin(matched_ids)
    ]

    if not matched.empty:
        train_s3_matches.append(matched)


if train_s3_matches:
    train_s3_matches = pd.concat(
        train_s3_matches,
        ignore_index=True
    )
else:
    train_s3_matches = pd.DataFrame()



# 7. Display each S1 together with its TRUE matches

for _, row in sample_gt.iterrows():

    s1_id = row["source1_entity_id"]

    print("\n")
    print("=" * 110)
    print("SOURCE 1 ENTITY:", s1_id)
    print("TRUE MATCHES:", row["matched_entity_ids"])
    print("=" * 110)


    # --------------------------------------------------------
    # Source 1 record
    # --------------------------------------------------------

    s1_record = train_s1_matches[
        train_s1_matches["entity_id"] == s1_id
    ]

    print("\nSOURCE 1:")

    if not s1_record.empty:

        print(
            s1_record[
                [
                    "entity_id",
                    "business_name",
                    "business_address",
                    "country"
                ]
            ].to_string(index=False)
        )

    else:

        print("S1 record not found.")


    
    # Get IDs belonging to this particular S1
    

    matched_ids_for_s1 = str(
        row["matched_entity_ids"]
    ).split(",")


    
    # Source 2 matches


    s2_records = train_s2_matches[
        train_s2_matches["entity_id"].isin(
            matched_ids_for_s1
        )
    ]

    print("\nSOURCE 2 MATCHES:")

    if not s2_records.empty:

        print(
            s2_records[
                [
                    "entity_id",
                    "business_name",
                    "business_address",
                    "country"
                ]
            ].to_string(index=False)
        )

    else:

        print("No Source 2 matches.")


    # --------------------------------------------------------
    # Source 3 matches
    # --------------------------------------------------------

    s3_records = train_s3_matches[
        train_s3_matches["entity_id"].isin(
            matched_ids_for_s1
        )
    ]

    print("\nSOURCE 3 MATCHES:")

    if not s3_records.empty:

        print(
            s3_records[
                [
                    "entity_id",
                    "business_name",
                    "business_address",
                    "country"
                ]
            ].to_string(index=False)
        )

    else:

        print("No Source 3 matches.")


print("\n\nFinished inspecting matched entities.")
# ============================================================
# STEP 4 — CREATE A CLEAN TRUE-MATCH COMPARISON TABLE
# ============================================================

comparison_rows = []

for _, row in sample_gt.iterrows():

    s1_id = row["source1_entity_id"]

    # Get the S1 record
    s1_record = train_s1_matches[
        train_s1_matches["entity_id"] == s1_id
    ]

    if s1_record.empty:
        continue

    s1_record = s1_record.iloc[0]

    # Get the true matched IDs for this S1
    matched_ids_for_s1 = str(
        row["matched_entity_ids"]
    ).split(",")

    # --------------------------------------------------------
    # Compare against S2 and S3 matches
    # --------------------------------------------------------

    for matched_id in matched_ids_for_s1:

        # Check Source 2
        s2_record = train_s2_matches[
            train_s2_matches["entity_id"] == matched_id
        ]

        if not s2_record.empty:

            matched_record = s2_record.iloc[0]

            comparison_rows.append({
                "s1_id": s1_id,
                "s1_name": s1_record["business_name"],
                "s1_address": s1_record["business_address"],
                "s1_country": s1_record["country"],

                "matched_source": "S2",
                "matched_id": matched_id,
                "matched_name": matched_record["business_name"],
                "matched_address": matched_record["business_address"],
                "matched_country": matched_record["country"]
            })

            continue

        # Check Source 3
        s3_record = train_s3_matches[
            train_s3_matches["entity_id"] == matched_id
        ]

        if not s3_record.empty:

            matched_record = s3_record.iloc[0]

            comparison_rows.append({
                "s1_id": s1_id,
                "s1_name": s1_record["business_name"],
                "s1_address": s1_record["business_address"],
                "s1_country": s1_record["country"],

                "matched_source": "S3",
                "matched_id": matched_id,
                "matched_name": matched_record["business_name"],
                "matched_address": matched_record["business_address"],
                "matched_country": matched_record["country"]
            })



# Convert the list into a DataFrame


comparison_df = pd.DataFrame(comparison_rows)

# Display basic information

print("\nComparison table shape:")
print(comparison_df.shape)

print("\nColumns:")
print(comparison_df.columns.tolist())

# Display the first 20 true matches


pd.set_option("display.max_colwidth", 80)

print("\nTRUE MATCHED PAIRS:")
print(comparison_df.head(20).to_string(index=False))

# Save the sample for easier inspection

comparison_df.to_csv(
    "true_matched_pairs_sample.csv",
    index=False
)

print("\nSaved as: true_matched_pairs_sample.csv")
import os

print(
    "File saved at:",
    os.path.abspath("true_matched_pairs_sample.csv")
)

# STEP 5 — BASIC MATCH QUALITY ANALYSIS


# 1. Exact name agreement
name_exact = (
    comparison_df["s1_name"].fillna("").str.lower().str.strip()
    ==
    comparison_df["matched_name"].fillna("").str.lower().str.strip()
)

print("Exact name matches:",
      name_exact.sum(),
      "/",
      len(comparison_df))

print(
    "Exact name match percentage:",
    name_exact.mean() * 100
)


# 2. Exact address agreement
address_exact = (
    comparison_df["s1_address"].fillna("").str.lower().str.strip()
    ==
    comparison_df["matched_address"].fillna("").str.lower().str.strip()
)

print(
    "\nExact address matches:",
    address_exact.sum(),
    "/",
    len(comparison_df)
)

print(
    "Exact address match percentage:",
    address_exact.mean() * 100
)


# 3. Country agreement
country_exact = (
    comparison_df["s1_country"].fillna("").str.lower().str.strip()
    ==
    comparison_df["matched_country"].fillna("").str.lower().str.strip()
)

print(
    "\nSame country:",
    country_exact.sum(),
    "/",
    len(comparison_df)
)

print(
    "Same country percentage:",
    country_exact.mean() * 100
)


# 4. Source distribution
print("\nMatched source distribution:")
print(comparison_df["matched_source"].value_counts())

# STEP 6 — NORMALIZATION EXPERIMENT


import re
import unicodedata


def normalize_basic(text):
    """
    Basic normalization for names/addresses.

    Operations:
    1. Handle missing values
    2. Normalize Unicode
    3. Convert to lowercase
    4. Replace '&' with 'and'
    5. Remove punctuation
    6. Normalize whitespace
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Treat '&' and 'and' consistently
    text = text.replace("&", " and ")

    # 4. Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 5. Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    # 6. Remove leading/trailing whitespace
    return text.strip()
comparison_df["s1_name_norm"] = (
    comparison_df["s1_name"]
    .apply(normalize_basic)
)

comparison_df["matched_name_norm"] = (
    comparison_df["matched_name"]
    .apply(normalize_basic)
)

comparison_df["s1_address_norm"] = (
    comparison_df["s1_address"]
    .apply(normalize_basic)
)

comparison_df["matched_address_norm"] = (
    comparison_df["matched_address"]
    .apply(normalize_basic)
)
name_exact_normalized = (
    comparison_df["s1_name_norm"]
    ==
    comparison_df["matched_name_norm"]
)

address_exact_normalized = (
    comparison_df["s1_address_norm"]
    ==
    comparison_df["matched_address_norm"]
)

print("\n========== NORMALIZATION RESULTS ==========")

print(
    "Normalized exact name matches:",
    name_exact_normalized.sum(),
    "/",
    len(comparison_df)
)

print(
    "Normalized name match percentage:",
    name_exact_normalized.mean() * 100
)

print(
    "\nNormalized exact address matches:",
    address_exact_normalized.sum(),
    "/",
    len(comparison_df)
)

print(
    "Normalized address match percentage:",
    address_exact_normalized.mean() * 100
)
improved_names = comparison_df[
    (comparison_df["s1_name"] != comparison_df["matched_name"])
    &
    (comparison_df["s1_name_norm"] == comparison_df["matched_name_norm"])
]

print("\nExamples where normalization made names equal:")

print(
    improved_names[
        [
            "s1_name",
            "matched_name",
            "s1_name_norm",
            "matched_name_norm"
        ]
    ].to_string(index=False)
)

# STEP 7 — LEGAL SUFFIX NORMALIZATION


LEGAL_SUFFIXES = [
    "private limited",
    "pvt limited",
    "pvt ltd",
    "private ltd",
    "limited",
    "ltd",
    "incorporated",
    "inc",
    "corporation",
    "corp",
    "company",
    "co",
    "llp",
    "llc",
]


def normalize_name_core(text):
    

    # First perform our basic normalization
    text = normalize_basic(text)

    if not text:
        return ""

    # Sort longest phrases first.
    # This prevents 'limited' from being removed before
    # 'private limited'.
    suffixes = sorted(
        LEGAL_SUFFIXES,
        key=len,
        reverse=True
    )

    # Remove a suffix only if it occurs at the END
    
    for suffix in suffixes:

        if text.endswith(" " + suffix):

            text = text[
                :-(len(suffix) + 1)
            ].strip()

            break

    return text
comparison_df["s1_name_core"] = (
    comparison_df["s1_name"]
    .apply(normalize_name_core)
)

comparison_df["matched_name_core"] = (
    comparison_df["matched_name"]
    .apply(normalize_name_core)
)
name_core_exact = (
    comparison_df["s1_name_core"]
    ==
    comparison_df["matched_name_core"]
)

print("\n========== LEGAL SUFFIX NORMALIZATION ==========")

print(
    "Core-name exact matches:",
    name_core_exact.sum(),
    "/",
    len(comparison_df)
)

print(
    "Core-name match percentage:",
    name_core_exact.mean() * 100
)
improved_core_names = comparison_df[
    (comparison_df["s1_name_norm"]
     != comparison_df["matched_name_norm"])
    &
    (comparison_df["s1_name_core"]
     == comparison_df["matched_name_core"])
]

print("\nExamples fixed by legal-suffix normalization:")

print(
    improved_core_names[
        [
            "s1_name",
            "matched_name",
            "s1_name_norm",
            "matched_name_norm",
            "s1_name_core",
            "matched_name_core"
        ]
    ].to_string(index=False)
)

# STEP 8 — ADDRESS ABBREVIATION NORMALIZATION


ADDRESS_REPLACEMENTS = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "lane": "ln",
    "highway": "hwy",
    "drive": "dr",
    "parkway": "pkwy",
    "place": "pl",
    "square": "sq",
}


def normalize_address(text):

    text = normalize_basic(text)

    if not text:
        return ""

    tokens = text.split()

    normalized_tokens = []

    for token in tokens:

        if token in ADDRESS_REPLACEMENTS:
            token = ADDRESS_REPLACEMENTS[token]

        normalized_tokens.append(token)

    return " ".join(normalized_tokens)
comparison_df["s1_address_norm2"] = (
    comparison_df["s1_address"]
    .apply(normalize_address)
)

comparison_df["matched_address_norm2"] = (
    comparison_df["matched_address"]
    .apply(normalize_address)
)
address_norm2_exact = (
    comparison_df["s1_address_norm2"]
    ==
    comparison_df["matched_address_norm2"]
)

print("\n========== ADDRESS NORMALIZATION ==========")

print(
    "Normalized address matches:",
    address_norm2_exact.sum(),
    "/",
    len(comparison_df)
)

print(
    "Normalized address percentage:",
    address_norm2_exact.mean() * 100
)

# FINAL EDA + NORMALIZATION SUMMARY


print("\n" + "=" * 70)
print("FINAL EDA + NORMALIZATION SUMMARY")
print("=" * 70)



# 1. Basic information about the dataset

print("\n[1] DATASET STRUCTURE")

print("Ground truth rows:", len(gt))
print("Sample matched pairs:", len(comparison_df))

print("\nMatch distribution:")
print(gt["num_matches"].value_counts().sort_index())

singleton_percentage = (
    (gt["num_matches"] == 0).mean() * 100
)

average_matches = gt["num_matches"].mean()
maximum_matches = gt["num_matches"].max()

print(
    "\nSingleton percentage:",
    round(singleton_percentage, 2),
    "%"
)

print(
    "Average matches per S1:",
    round(average_matches, 2)
)

print(
    "Maximum matches per S1:",
    maximum_matches
)



# 2. Where are the matches coming from?


print("\n[2] MATCH SOURCE DISTRIBUTION")

print(
    comparison_df["matched_source"].value_counts()
)



# 3. Raw matching baseline


raw_name_match = (
    comparison_df["s1_name"]
    .fillna("")
    .str.lower()
    .str.strip()
    ==
    comparison_df["matched_name"]
    .fillna("")
    .str.lower()
    .str.strip()
)

raw_address_match = (
    comparison_df["s1_address"]
    .fillna("")
    .str.lower()
    .str.strip()
    ==
    comparison_df["matched_address"]
    .fillna("")
    .str.lower()
    .str.strip()
)



# 4. Results after basic normalization


basic_name_match = (
    comparison_df["s1_name_norm"]
    ==
    comparison_df["matched_name_norm"]
)

basic_address_match = (
    comparison_df["s1_address_norm"]
    ==
    comparison_df["matched_address_norm"]
)



# 5. Results after removing common legal suffixes


core_name_match = (
    comparison_df["s1_name_core"]
    ==
    comparison_df["matched_name_core"]
)



# 6. Results after address abbreviation normalization


address_norm2_match = ( comparison_df["s1_address_norm2"]== comparison_df["matched_address_norm2"] )
    
# 7. Compare all normalization approaches


print("\n[3] NORMALIZATION EFFECT")

print(
    f"Raw exact name:              "
    f"{raw_name_match.sum()}/{len(comparison_df)} "
    f"({raw_name_match.mean() * 100:.2f}%)"
)

print(
    f"Basic normalized name:       "
    f"{basic_name_match.sum()}/{len(comparison_df)} "
    f"({basic_name_match.mean() * 100:.2f}%)"
)

print(
    f"Core normalized name:        "
    f"{core_name_match.sum()}/{len(comparison_df)} "
    f"({core_name_match.mean() * 100:.2f}%)"
)

print()

print(
    f"Raw exact address:            "
    f"{raw_address_match.sum()}/{len(comparison_df)} "
    f"({raw_address_match.mean() * 100:.2f}%)"
)

print(
    f"Basic normalized address:     "
    f"{basic_address_match.sum()}/{len(comparison_df)} "
    f"({basic_address_match.mean() * 100:.2f}%)"
)

print(
    f"Abbreviation normalized addr: "
    f"{address_norm2_match.sum()}/{len(comparison_df)} "
    f"({address_norm2_match.mean() * 100:.2f}%)"
)

# 8. Check whether the country agrees

country_match = (
    comparison_df["s1_country"]
    .fillna("")
    .str.lower()
    .str.strip()
    ==
    comparison_df["matched_country"]
    .fillna("")
    .str.lower()
    .str.strip()
)

print(
    f"\nCountry agreement:            "
    f"{country_match.sum()}/{len(comparison_df)} "
    f"({country_match.mean() * 100:.2f}%)"
)

# 9. Check missing values in the sample

print("\n[4] MISSING VALUES IN MATCH SAMPLE")

important_columns = [
    "s1_name",
    "s1_address",
    "s1_country",
    "matched_name",
    "matched_address",
    "matched_country"
]

for column in important_columns:

    missing_count = comparison_df[column].isna().sum()

    print(
        f"{column:20} : "
        f"{missing_count}/{len(comparison_df)} missing "
        f"({missing_count / len(comparison_df) * 100:.2f}%)"
    )



# 10. Look at business-name lengths


print("\n[5] NAME LENGTH STATISTICS")

comparison_df["s1_name_length"] = (
    comparison_df["s1_name_norm"].str.len()
)

comparison_df["matched_name_length"] = (
    comparison_df["matched_name_norm"].str.len()
)

print(
    "S1 average name length:",
    round(comparison_df["s1_name_length"].mean(), 2)
)

print(
    "Matched average name length:",
    round(
        comparison_df["matched_name_length"].mean(),
        2
    )
)

# 11. Look at address lengths


comparison_df["s1_address_length"] = (
    comparison_df["s1_address_norm"].str.len()
)

comparison_df["matched_address_length"] = (
    comparison_df["matched_address_norm"].str.len()
)

print("\n[6] ADDRESS LENGTH STATISTICS")

print("S1 average address length:",round(comparison_df["s1_address_length"].mean(),2)
)

print("Matched average address length:",round(comparison_df["matched_address_length"].mean(),2)
)

# 12. Save the normalized sample

output_columns = [
    "s1_id",
    "s1_name",
    "s1_name_norm",
    "s1_name_core",
    "s1_address",
    "s1_address_norm",
    "s1_address_norm2",
    "s1_country",

    "matched_source",
    "matched_id",
    "matched_name",
    "matched_name_norm",
    "matched_name_core",
    "matched_address",
    "matched_address_norm",
    "matched_address_norm2",
    "matched_country"
]

comparison_df[output_columns].to_csv(
    "normalized_matched_pairs_sample.csv",
    index=False
)

# 13. Save all the important results in one summary file

summary = {
    "total_ground_truth_rows": len(gt),
    "sample_true_matches": len(comparison_df),

    "singleton_percentage":
        singleton_percentage,

    "average_matches_per_s1":
        average_matches,

    "maximum_matches_per_s1":
        maximum_matches,

    "raw_name_match_percentage":
        raw_name_match.mean() * 100,

    "basic_name_match_percentage":
        basic_name_match.mean() * 100,

    "core_name_match_percentage":
        core_name_match.mean() * 100,

    "raw_address_match_percentage":
        raw_address_match.mean() * 100,

    "basic_address_match_percentage":
        basic_address_match.mean() * 100,

    "normalized_address_match_percentage":
        address_norm2_match.mean() * 100,

    "country_match_percentage":
        country_match.mean() * 100
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    "eda_normalization_summary.csv",
    index=False
)


