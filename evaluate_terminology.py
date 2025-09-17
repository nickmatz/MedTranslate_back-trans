"""
Script written by Nick Matzopoulos (mtznic006@myuct.ac.za)

Used to calculate the health term error rate of a model by taking in it's
predictions, the reference file and the health terms to check and calculating how many 
of the terms were translated correctly.

This is output as a percentage in four categories: Overall, Anatomy, Condition, Treatment.
"""

import argparse
import os
import sys

try:
    import pandas as pd
    from collections import defaultdict
except ImportError:
    print("ERROR: The 'pandas' library is required but not installed.", file=sys.stderr)
    print("Please install it by running: pip3 install pandas", file=sys.stderr)
    sys.exit(1)



def load_terminology(path):
    """Loads the medical terminology CSV into a structured dictionary."""
    df = pd.read_csv(path)
    terminology = defaultdict(lambda: {'en': set(), 'xh': set()})
    for _, row in df.iterrows():
        category = row['Category'].strip().lower()
        en_term = row['English'].strip().lower()
        # Handle potentially empty isiXhosa cells
        xh_term_raw = row.get('isiXhosa')
        xh_term = str(xh_term_raw).strip().lower() if pd.notna(xh_term_raw) else ""
        if xh_term.startswith('-'):
            xh_term = xh_term[1:]
        if en_term: terminology[category]['en'].add(en_term)
        if xh_term: terminology[category]['xh'].add(xh_term)
    return dict(terminology)

def read_text_file(path):
    """Reads a text file and returns its content as a single lowercase string."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().lower()

def calculate_error_rate(predictions_text, references_text, term_list):
    """Calculates the error rate for a given list of terms."""
    total_in_reference = 0
    correct_in_prediction = 0

    if not term_list:
        return 0.0, 0, 0

    for term in term_list:
        count_in_ref = references_text.count(term)
        count_in_pred = predictions_text.count(term)
        total_in_reference += count_in_ref
        correct_in_prediction += min(count_in_ref, count_in_pred)

    if total_in_reference == 0:
        return 0.0, 0, 0

    error_rate = 1.0 - (correct_in_prediction / total_in_reference)
    return error_rate * 100, correct_in_prediction, total_in_reference

def main():
    parser = argparse.ArgumentParser(description="Evaluate health term translation accuracy.")
    parser.add_argument('--predictions', type=str, required=True, help="Path to the model's generated translation file.")
    parser.add_argument('--references', type=str, required=True, help="Path to the human reference translation file.")
    parser.add_argument('--terms_csv', type=str, required=True, help="Path to the medical terms CSV file.")
    parser.add_argument('--direction', type=str, required=True, choices=['en_to_xh', 'xh_to_en'], help="Translation direction.")
    args = parser.parse_args()

    print("--- Script Started ---")
    # --- File existence checks for better error messages ---
    for file_path in [args.predictions, args.references, args.terms_csv]:
        if not os.path.exists(file_path):
            print(f"FATAL ERROR: Input file not found at '{file_path}'", file=sys.stderr)
            sys.exit(1)

    try:
        terminology = load_terminology(args.terms_csv)
        print(f"Terminology loaded from {args.terms_csv}")
        predictions_text = read_text_file(args.predictions)
        print(f"Predictions loaded from {args.predictions}")

        references_text = read_text_file(args.references)
        print(f"References loaded from {args.references}")

    except Exception as e:
        print(f"FATAL ERROR: Failed to load or process input files: {e}", file=sys.stderr)
        sys.exit(1)

    lang_key = 'xh' if args.direction == 'en_to_xh' else 'en'
    results = {}
    all_terms = set()
    categories = ['anatomy', 'condition', 'treatment']
    for category in categories:
        term_list = terminology.get(category, {}).get(lang_key, set())
        all_terms.update(term_list)
        error_rate, _, _ = calculate_error_rate(predictions_text, references_text, term_list)
        results[category.capitalize()] = f"{error_rate:.2f}%"

    overall_error_rate, total_correct, total_possible = calculate_error_rate(predictions_text, references_text, all_terms)
    results['Overall'] = f"{overall_error_rate:.2f}%"
    print("\n--- Health Term Error Rate ---")
    print(f"Direction: {args.direction.replace('_', ' -> ').upper()}")
    print("-" * 50)
    results_df = pd.DataFrame([results], columns=['Overall', 'Anatomy', 'Condition', 'Treatment'])
    print(results_df.to_string(index=False))
    print("-" * 50)
    print(f"Summary: Correctly translated {total_correct} out of {total_possible} medical terms.")

if __name__ == "__main__":
    main() 