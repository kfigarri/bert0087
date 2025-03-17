#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from collections import Counter
from datetime import datetime
from typing import List, Tuple, Callable, Dict, Optional

import textstat
import nltk
from nltk.corpus import stopwords
from pydantic import BaseModel
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

# Data models.
class QASnippet(BaseModel):
    file_path: str
    span: Tuple[int, int]
    answer: str

class QAGroundTruth(BaseModel):
    query: str
    snippets: List[QASnippet]
    file_set: Optional[List[str]] = None

def load_groundtruth(json_file_path: str) -> List[QAGroundTruth]:
    """
    Loads the QA ground-truth data from a JSON file.
    Expected JSON format:
    [
        {
            "query": "Your query...",
            "snippets": [
                { "file_path": "path/to/file.txt", "span": [start, end], "answer": "The answer text..." },
                ...
            ],
            "file_set": [ "file1.txt", "file2.txt", ... ]
        },
        ...
    ]
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    groundtruth_tests = []
    tests = data.get("tests") if isinstance(data, dict) else data
    for test in tests:
        snippets = [QASnippet(**snippet) for snippet in test.get("snippets", [])]
        file_set = test.get("file_set")
        groundtruth_tests.append(QAGroundTruth(query=test["query"], snippets=snippets, file_set=file_set))
    return groundtruth_tests

def calculate_readability(query: str) -> str:
    """
    Calculate different readability scores based on Dale Chall and categories into expert and non-expert based on a specific treshold
    """
    dale_chall = textstat.dale_chall_readability_score(query)

    if dale_chall < 8.0 :
        return "non-expert", dale_chall
    else:
        return "expert", dale_chall

def split_question_ner(query: str, ner_model) -> Tuple[str, str]:
    """
    Splits a query into two parts using NER.
    
    If the query contains a semicolon, it splits into:
      - targeted_corpus: text before semicolon (after removing "Consider")
      - original_question: text after semicolon.
    
    Otherwise, uses the NER pipeline to extract ORG or MISC entities (with score > 0.8),
    removes unwanted words, and joins all remaining entities as targeted_corpus.
    The remainder (with the targeted_corpus removed) is taken as the original question.
    """
    unwanted_words = ['agreement', 'nda', 'non-disclosure', 'agreements', 'content', 'co-branding', 'license', "acquisition", "merger"]
    pattern = r"^Consider (.*?);"
    match = re.match(pattern, query)
    if match:
        tgt = match.group(1).strip()
        tgt = re.sub(r"(?i)Non-Disclosure Agreement", "", tgt).strip()
        stop_words = set(stopwords.words("english"))
        tokens = tgt.split()
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        targeted_corpus = " ".join(filtered_tokens)
        parts = query.split(";", 1)
        original_question = parts[1].strip() if len(parts) > 1 else ""
        return targeted_corpus, original_question
    else:
        ner_results = ner_model(query)
        org_entities = [ent["word"].strip() for ent in ner_results if ent.get("entity_group") in ["ORG", "MISC"] and ent.get("score", 0) > 0.8]
        pattern = re.compile(r"\b(" + "|".join(unwanted_words) + r")\b", re.IGNORECASE)
        filtered_orgs = [pattern.sub("", org).strip() for org in org_entities]
        filtered_orgs = [org for org in filtered_orgs if org]
        if filtered_orgs:
            targeted_corpus = " ".join(filtered_orgs)
        else:
            targeted_corpus = ""
        original_question = query.replace(targeted_corpus, "").strip()
        return targeted_corpus, original_question

def find_best_corpus_embeddings(tgt_corpus: str, corpus_files: List[str],
                                model: SentenceTransformer) -> Tuple[str, float]:
    """
    Embeds the target corpus description and each file name using a sentence transformer,
    then computes cosine similarities to find the best matching file.
    """
    tgt_embedding = model.encode(tgt_corpus, convert_to_tensor=True)
    file_embeddings = model.encode(corpus_files, convert_to_tensor=True)
    cosine_scores = util.cos_sim(tgt_embedding, file_embeddings)[0]
    best_idx = int(cosine_scores.argmax())
    best_score = float(cosine_scores[best_idx])
    return corpus_files[best_idx], best_score

def query_rewriter(
    ground_truths: List[QAGroundTruth],
    candidate_files: List[str],
    threshold: float,
    match_fn: Callable[[str, List[str]], Tuple[str, float]],
    split_fn: Callable[[str], Tuple[str, str]]
) -> List[Dict]:
    """
    For each QAGroundTruth:
      - Splits the query using split_fn into:
          * targeted_corpus (the extracted document or agreement details)
          * only_question (the actual question about that document)
      - Finds the best matching file using match_fn and its similarity score.
      - Computes a simple query complexity metric and corresponding retrieved_k value.
      - Determines a score:
            * If similarity >= threshold: score is 1 if best_file is among the ground truth's actual file paths, else -1.
            * Otherwise, score is 0.
    
    Returns a list of dictionaries in the new format with keys:
      - "query", "snippets", "file_set" (if available),
      - "query_rewriter": a list with a dictionary containing:
            "best_file_path", "file_locator", "similarity_score", "only_question"
      - "feature_extraction": a list with a dictionary containing:
            "query_complexity", "retrieved_k"
    """
    outputs = []
    for gt in tqdm(ground_truths, desc="Evaluating queries"):
        targeted_corpus, only_question = split_fn(gt.query)
        best_file, similarity = match_fn(targeted_corpus, candidate_files)
        
        if similarity >= threshold:
            query_rewriter_value = {
                "best_file_path": best_file,
                "file_locator": targeted_corpus,
                "similarity_score": similarity,
                "only_question": only_question
            }
            readability, readability_score = calculate_readability(only_question)
        else:
            query_rewriter_value = {
                "best_file_path": "",
                "file_locator": targeted_corpus,
                "similarity_score": similarity,
                "only_question": ""
            }
            readability, readability_score = calculate_readability(gt.query)
        
        result = {
            "query": gt.query,
            "snippets": [snippet.dict() for snippet in gt.snippets],
            "file_set": gt.file_set if gt.file_set is not None else [],
            "query_rewriter": [query_rewriter_value],
            "feature_extraction": [
                {
                    "complexity": "",
                    "readability": readability,
                    "readability_score": readability_score
                }
            ],
        }
        outputs.append(result)
    return outputs

def save_evaluation_results(results: List[Dict], output_file: str) -> None:
    """
    Saves the evaluation results (a list of dictionaries) to a JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {output_file}")

def main(args):
    # Set parameters from command-line arguments.
    _use_sample = args.use_sample
    dataset_type = args.dataset_type
    dataset_name = args.dataset_name

    vectorstore_path = f"./vectorstore/faiss_store_{'sample_' if _use_sample else ''}{dataset_name}"
    directory_path = f"../data/{'sample_' if _use_sample else ''}corpus/{dataset_name}"
    test_file = f"../data/{'sample_' if _use_sample else ''}benchmarks/{dataset_name}.json"
    rephrased_file = f"../data/{'sample_' if _use_sample else ''}benchmarks_rephrased/{dataset_name}.json"
    generic_file = f"../data/{'sample_' if _use_sample else ''}benchmarks_generic/{dataset_name}.json"
    
    if dataset_type == "rephrased":
        curr_file = rephrased_file
        prefix_dataset_type = "_rephrased"
    elif dataset_type == "generic":
        curr_file = generic_file
        prefix_dataset_type = "_generic"
    else:
        curr_file = test_file
        prefix_dataset_type = ""

    if dataset_name == "cuad":
        threshold = 0.55
    elif dataset_name == "maud":
        threshold = 0.38
    else:
        threshold = 0.3
    
    print(f"use sample: {_use_sample}")
    print(f"dataset: {dataset_name + prefix_dataset_type}")
    print(f"vector store: {vectorstore_path}")
    print(f"corpus path: {directory_path}")
    print(f"test file: {test_file}")
    print(f"rephrased file: {rephrased_file}")
    print(f"generic file: {generic_file}")
    
    groundtruth_tests = load_groundtruth(curr_file)
    list_corpus = [os.path.join(f"{dataset_name}", filename) for filename in os.listdir(directory_path) if filename.endswith(".txt")]
    
    # Use SentenceTransformer for matching.
    model_embed = SentenceTransformer("all-MiniLM-L6-v2")
    match_fn_embeddings = lambda tgt, files: find_best_corpus_embeddings(tgt, files, model_embed)
    
    # Use a NER model for splitting.
    ner_model = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", aggregation_strategy="simple")
    split_fn = lambda query: split_question_ner(query, ner_model)
    
    results = query_rewriter(groundtruth_tests, list_corpus, threshold, match_fn_embeddings, split_fn)
    
    for sample in random.sample(results, 1):
        print(json.dumps(sample, indent=2))
    
    output_file = f"../data/results/query_translation/{'all_' if not _use_sample else ''}{dataset_name}{prefix_dataset_type}.json"
    save_evaluation_results(results, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Translation Evaluation")
    parser.add_argument("--use_sample", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="_use_sample: True/False")
    parser.add_argument("--dataset_type", type=str, choices=["", "rephrased", "generic"], default="",
                        help="dataset_type: '', 'rephrased', or 'generic'")
    parser.add_argument("--dataset_name", type=str, choices=["contractnli", "cuad", "maud", "privacy_qa"], default="contractnli",
                        help="dataset_name: one of 'contractnli', 'cuad', 'maud', 'privacy_qa'")
    
    args = parser.parse_args()
    main(args)