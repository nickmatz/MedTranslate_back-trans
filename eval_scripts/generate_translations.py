"""
Script written by Nick Matzopoulos (mtznic006@myuct.ac.za)

This program is used to generate translations of the given source file to compare to the 
reference file for evaluation. 
"""

import os
import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import argparse
from tqdm import tqdm
from safetensors.torch import load_file

def main():
    parser = argparse.ArgumentParser(description="Generate translations using the given model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model directory.")
    parser.add_argument('--source_file', type=str, required=True, help="Path to the source text file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the generated translations.")
    parser.add_argument('--source_lang', type=str, required=True, help="Source language code (e.g., eng_Latn).")
    parser.add_argument('--target_lang', type=str, required=True, help="Target language code (e.g., xho_Latn).")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for generation.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    print(f"Loading model from: {args.model_path}...")
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_config(config)
    safetensors_path = os.path.join(args.model_path, "model.safetensors")
    state_dict = load_file(safetensors_path, device="cpu")
    

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    print("Model and tokenizer loaded.")

    tokenizer.src_lang = args.source_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(args.target_lang)

    with open(args.source_file, 'r', encoding='utf-8') as f:
        source_texts = [line.strip() for line in f]
    print(f"Loaded {len(source_texts)} sentences from {args.source_file}")

    model.eval()
    generated_translations = []
    print("Starting translation generation...")
    for i in tqdm(range(0, len(source_texts), args.batch_size)):
        batch = source_texts[i:i + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=5
            )
        
        translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        generated_translations.extend(translations)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for line in generated_translations:
            f.write(line + '\n')
            
    print(f"\nSuccessfully generated {len(generated_translations)} translations.")
    print(f"Saved to: {args.output_file}")

if __name__ == "__main__":
    main()