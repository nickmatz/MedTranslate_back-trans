# filename: create_training_files.py
# A flexible script to convert a translated CSV into parallel training files.

import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Create parallel training files from a translated CSV.")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing translations.')
    parser.add_argument('--output_prefix', type=str, required=True, help='The base name for the output files (e.g., "47_train", "back_train").')
    parser.add_argument('--output_dir', type=str, default='../finetune_data', help='Directory to save the output files.')
    args = parser.parse_args()

    print("--- Starting data preparation ---")
    print(f"Reading from: {args.input_csv}")
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct the full output paths
    output_en_path = os.path.join(args.output_dir, f"{args.output_prefix}.en")
    output_xh_path = os.path.join(args.output_dir, f"{args.output_prefix}.xh")
    
    print(f"Will save English text to: {output_en_path}")
    print(f"Will save isiXhosa text to: {output_xh_path}")

    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"FATAL ERROR: The input file was not found at '{args.input_csv}'")
        return

    # Check for the expected columns
    if 'text' not in df.columns or 'isixhosa_translation' not in df.columns:
        print("FATAL ERROR: The CSV must contain 'text' and 'isixhosa_translation' columns.")
        return

    # Create parallel text files
    with open(output_en_path, 'w', encoding='utf-8') as f_en, \
         open(output_xh_path, 'w', encoding='utf-8') as f_xh:
        
        count = 0
        for _, row in df.iterrows():
            # Ensure both the English and isiXhosa text are not empty/NaN
            if pd.notna(row['text']) and pd.notna(row['isixhosa_translation']):
                # Clean up text: remove newlines and extra spaces
                eng_text = ' '.join(str(row['text']).split())
                xho_text = ' '.join(str(row['isixhosa_translation']).split())
                
                if eng_text and xho_text:
                    f_en.write(eng_text + '\n')
                    f_xh.write(xho_text + '\n')
                    count += 1
    
    print(f"\n✓ Processing complete. Wrote {count} parallel sentences.")

if __name__ == "__main__":
    main()