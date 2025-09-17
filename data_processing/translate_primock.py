# filename: translate_primock.py
# Written by Nick Matzopoulos

import torch
import pandas as pd
import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse

def translate_batch(texts, model, tokenizer, src_lang="eng_Latn", tgt_lang="xho_Latn", device="cuda"):
    """Translates a batch of texts using the NLLB model."""
    tokenizer.src_lang = src_lang
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    # Use the target language code directly, as NLLB tokenizers are built for this
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tgt_lang_id,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Translate a CSV file using NLLB-200-1.3B model.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output CSV file.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for translation.')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("Warning: Running on CPU will be very slow.")

    model_path = "/mnt/lustre/users/nmatzopoulos/nmt_project/hf_cache_final/hub/nllb-200-1.3B"
    print(f"Loading model from direct local path: {model_path}...")
    
    try:
       
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
    except Exception as e:
        print(f"Standard from_pretrained failed: {e}")
        print("Falling back to robust manual loading...")
        
       
        
        # Step 1: Load config
        config = AutoConfig.from_pretrained(model_path)
        
        # Step 2: Create model on CPU (not meta device)
        model = AutoModelForSeq2SeqLM.from_config(config)
        
        # Step 3: Find the weights file, whether it's .safetensors or .bin
        weights_file = None
        # The list now includes the file you actually have
        for filename in ["model.safetensors", "pytorch_model.bin"]:
            potential_path = os.path.join(model_path, filename)
            if os.path.exists(potential_path):
                weights_file = potential_path
                break
        
        if weights_file is None:
            
            print(f"Contents of {model_path}:")
            for file in os.listdir(model_path): print(f"  - {file}")
            raise FileNotFoundError(f"Could not find a weights file (model.safetensors or pytorch_model.bin) in the model directory.")
        
        print(f"Found weights file: {weights_file}")
        
        # Step 4: Load the weights based on the file type
        if weights_file.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(weights_file, device="cpu")
        else: # It must be a .bin file
            
            state_dict = torch.load(weights_file, map_location="cpu")

        # Load the state dictionary into the model architecture
        model.load_state_dict(state_dict, strict=False)
        
        # Step 5: Move to the correct device
        model = model.to(device)
        
        # Load tokenizer normally
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    print("Model loaded successfully to GPU.")
    
    print(f"Reading input file: {args.input_file}")
    df = pd.read_csv(args.input_file)
    texts_to_translate = df['text'].tolist()

    all_translations = []
    print(f"Starting translation with batch size {args.batch_size}...")
    for i in tqdm(range(0, len(texts_to_translate), args.batch_size)):
        batch_texts = texts_to_translate[i:i + args.batch_size]
        non_empty_texts = [text for text in batch_texts if isinstance(text, str) and text.strip() != ""]
        if not non_empty_texts:
            batch_translations = [''] * len(batch_texts)
        else:
            translated = translate_batch(non_empty_texts, model, tokenizer, device=device)
            translated_iter = iter(translated)
            batch_translations = [next(translated_iter) if isinstance(text, str) and text.strip() != "" else "" for text in batch_texts]
        all_translations.extend(batch_translations)

    df['isixhosa_translation'] = all_translations
    
    print(f"Translation complete. Saving results to: {args.output_file}")
    df.to_csv(args.output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    main()