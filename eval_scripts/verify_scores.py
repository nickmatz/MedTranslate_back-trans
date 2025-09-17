"""
Script written by Nick Matzopoulos (mtznic006@myuct.ac.za)

Program used to manually calculate BLEU, chrF and chrF++ scores. This is done by giving the predictions
made by a model and the reference translations.
"""

import sacrebleu
import argparse

def read_text_file(path):
    """Reads a text file using .read().splitlines()"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return lines

def main():
    parser = argparse.ArgumentParser(description="Manually calculate BLEU, chrF, and chrF++ scores.")
    parser.add_argument('--predictions', type=str, required=True, help="Path to the file with model predictions.")
    parser.add_argument('--references', type=str, required=True, help="Path to the file with reference translations.")
    args = parser.parse_args()

    print(f"Loading predictions from: {args.predictions}")
    print(f"Loading references from: {args.references}")

    predictions = read_text_file(args.predictions)
    references = [read_text_file(args.references)]

    # Using the 'intl' tokenizer as it is better for isiXhosa evaluation
    bleu = sacrebleu.corpus_bleu(predictions, references, tokenize='intl') 
    chrf = sacrebleu.corpus_chrf(predictions, references)
    chrf_plus = sacrebleu.corpus_chrf(predictions, references, word_order=2)

    print("\n--- VERIFICATION RESULTS ---")
    print(f"Number of sentences: {len(predictions)}")
    print(f"BLEU: {bleu.score:.2f}")
    print(f"chrF: {chrf.score:.2f}")
    print(f"chrF++: {chrf_plus.score:.2f}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()