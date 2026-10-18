# Dataset

`dataset_with_test_fold.csv` contains the 1,793 requirements used in all experiments.

| Column | Meaning |
|---|---|
| `id` | Requirement id (`<original split>__<hash>`) |
| `text` | Requirement text |
| `L1` | ISO/IEC 25010 characteristic (7 classes) |
| `L2` | ISO/IEC 25010 sub-characteristic (24 classes) |
| `source_split`, `orig_id` | Split and id in the original FR_NFR release |
| `cv_index` | Row number used by the original experiment scripts |
| `test_fold` | Fixed cross-validation fold (0–4) |

Fold sizes: 359, 359, 359, 358 and 358 requirements. When fold *k* is the test fold, fold (*k* + 1) mod 5 is the validation fold and the remaining three folds are used for training.

Level-1 distribution: security 450, usability 354, performance efficiency 353, reliability 295, portability 128, maintainability 116, compatibility 97.

## Source and construction

The source is the FR_NFR dataset ([Mendeley Data](https://data.mendeley.com/datasets/4ysx9fyzv4/1)). The dataset page reports 6,118 requirements (3,964 FR and 2,154 NFR); the released file contains 6,117 requirements: 3,964 FR, 2,122 NFR and 31 without a type label. All 2,122 NFR-labelled requirements were annotated by three annotators, and 329 were excluded:

| Reason | Details | Removed |
|---|---|---|
| Duplicates | Exact copies (57) and copies differing only in spacing (2) | 59 |
| Vague or invalid | Not an NFR (6), multi-concern (4), constraint (3), recommendation (2), vague (2), process statement (2) | 19 |
| Functional suitability | Appropriateness (73), correctness (21), completeness (16) | 110 |
| Minority characteristics (≤ 30 each, outside the product quality model) | Regulation compliance (30), freedom from risk (24), satisfaction (19), data privacy (17), tailorability (15), context coverage (9), efficiency (5), robustness (5), timeliness (5), effectiveness (3) | 132 |
| Minority sub-characteristics (< 10 each) | Co-existence (5), reusability (4) | 9 |
| **Total** | | **329** |

## License

The dataset and label cards are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The requirement texts come from the FR_NFR dataset (Mendeley Data, DOI: [10.17632/4ysx9fyzv4.1](https://doi.org/10.17632/4ysx9fyzv4.1)), also licensed under CC BY 4.0; please cite that source as well.
