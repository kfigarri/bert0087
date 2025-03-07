
# Query Translation Evaluation

This project evaluates query translation by splitting ground truth queries into two parts—one that describes the relevant document (or agreement) and one that contains the actual query—and then evaluates how well the extracted information matches the expected corpus file.

## Overview

The `query_translation.py` script:
- Loads ground truth test data from a JSON file.
- Splits each query into two parts using a Named Entity Recognition (NER) or regex-based method.
- Finds the best matching corpus file for the extracted targeted corpus using a matching function (e.g. via sentence embeddings or RapidFuzz).
- Computes a simple query complexity metric and assigns a score based on the similarity match.
- Saves the evaluation results in a structured JSON format with details such as:
  - Original query.
  - Extracted `targeted_corpus` and `only_question`.
  - The best matching file path and similarity score.
  - A computed `query_complexity` and corresponding `retrieved_k` value.
  - A final score indicating whether the matching was successful.
  
The script supports command-line arguments so that you can adjust the dataset parameters and use sample data if desired.

## Prerequisites

- Python 3.8 or later.
- Required packages:
  - `transformers`
  - `sentence_transformers`
  - `openai` (if using OpenAI API; not required for the current version)
  - `nltk`
  - `tqdm`
  - `pydantic`
  
Install dependencies via pip (for example):

```bash
pip install transformers sentence_transformers openai nltk tqdm pydantic
```

Additionally, download the NLTK stopwords (the script does this automatically, but you can also run):

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

Make sure your `query_translation.py` file is executable and then run it from the command line. The script accepts the following command-line arguments:

- `--use_sample`:  
  Boolean flag to indicate whether to use sample data.  
  **Default:** `True`  
  **Possible values:** `True` or `False`

- `--dataset_type`:  
  Type of dataset to load.  
  **Default:** `""` (empty string)  
  **Possible values:** `""`, `"rephrased"`, or `"generic"`

- `--dataset_name`:  
  Name of the dataset.  
  **Default:** `"privacy_qa"`  
  **Possible values:** `"contractnli"`, `"cuad"`, `"maud"`, `"privacy_qa"`

- `--threshold`:  
  A float value that defines the minimum similarity threshold for matching.  
  **Default:** `0.3`

### Example Command

```bash
python3 query_translation.py --use_sample True --dataset_type rephrased --dataset_name privacy_qa --threshold 0.3
```

When you run the script, it will:
- Print the current configuration (sample usage, dataset name, file paths, etc.).
- Load the ground truth data from the appropriate JSON file.
- Build a candidate file list from the corpus directory.
- Evaluate the queries using the specified splitting and matching functions.
- Print some random sample outputs and a score distribution.
- Save the final evaluation results in a JSON file under the `../data/results/query_translation/` directory, with a timestamp in the filename.

## Customization

- **Splitting Method:**  
  The script uses a NER-based splitting function (`split_question_ner`). You can modify or replace it with your own splitting logic if needed.

- **Matching Function:**  
  The current implementation uses sentence embeddings via `all-MiniLM-L6-v2` to compute similarity. You can swap this out with other matching functions (for example, using RapidFuzz).

- **Query Complexity:**  
  The `compute_query_complexity()` function currently returns a random complexity label and retrieved_k value. You can update this logic to compute complexity based on query length or other features.

## License

This project is open source and available under the MIT License.