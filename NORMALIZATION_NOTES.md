# Normalization Notes

## Why normalization was tested

Exact string matching performs poorly because the same business can appear differently across the three sources.

## Observed baseline

Measured on the checked-in sample of 33 known true matched pairs:

- Raw exact name matching: 12.12%
- Raw exact address matching: 3.03%

## Name normalization

Basic normalization includes:
- Unicode compatibility normalization
- Lowercase conversion
- Punctuation handling
- Whitespace normalization
- `&` replaced with `and`

Core-name normalization additionally removes common legal suffixes when they occur at the end of a business name.

Example:

```text
ABC Pharma Private Limited
→ abc pharma private limited
→ abc pharma
```

## Address normalization

Basic normalization includes:
- Unicode compatibility normalization
- Lowercase conversion
- Punctuation handling
- Whitespace normalization

Address abbreviation normalization includes mappings such as:

```text
road → rd
street → st
avenue → ave
boulevard → blvd
lane → ln
highway → hwy
drive → dr
```

## Experimental result

On the 33-pair sample, core-name normalization increased exact name agreement from 12.12% to 39.39%. Address normalization increased exact address agreement from 3.03% to 6.06%.

## Caution

These results are based on a small sample of known true matches. Do not assume these percentages represent the entire dataset. Treat normalized fields as matching-model features, not as the final matching decision, and validate on a larger training sample.
