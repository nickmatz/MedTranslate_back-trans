"""
Written by Nick Matzopoulos (mtznic006@myuct.ac.za)

This program is used to fine-tune a model for isiXhosa to English translation.
"""

import os
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import argparse
import sacrebleu
from safetensors.torch import load_file

class M2M100Trainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to handle M2M100's specific requirements
        """
        # Remove decoder_inputs_embeds if it exists
        if "decoder_inputs_embeds" in inputs:
            inputs.pop("decoder_inputs_embeds")
        
        # Ensure we only have decoder_input_ids
        if "labels" in inputs:
            labels = inputs.pop("labels")
            if "decoder_input_ids" not in inputs:
                # Shift labels to create decoder_input_ids
                decoder_input_ids = labels.new_zeros(labels.shape)
                decoder_input_ids[..., 1:] = labels[..., :-1].clone()
                decoder_input_ids[..., 0] = self.model.config.decoder_start_token_id
                decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.tokenizer.pad_token_id)
                inputs["decoder_input_ids"] = decoder_input_ids
            inputs["labels"] = labels
        
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

def fix_model_state_dict(state_dict, model):
    """Fix state dict keys to match model structure"""
    print("\n--- Fixing state dict keys ---")
    
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    
    fixed_state_dict = {}
    
    # Copy all keys as-is
    for key, value in state_dict.items():
        fixed_state_dict[key] = value
    
    # Fix missing keys
    missing_keys = model_keys - loaded_keys
    print(f"Missing keys: {len(missing_keys)}")
    
    if missing_keys:
        print("Attempting to map keys...")
        
        for missing_key in missing_keys:
            if missing_key == 'model.shared.weight':
                for k in ['model.decoder.embed_tokens.weight', 'decoder.embed_tokens.weight']:
                    if k in fixed_state_dict:
                        fixed_state_dict[missing_key] = fixed_state_dict[k]
                        print(f"Mapped shared weight from {k}")
                        break
            
            elif 'embed_tokens' in missing_key:
                if 'encoder.embed_tokens.weight' in missing_key:
                    for k, v in state_dict.items():
                        if 'embed_tokens.weight' in k:
                            fixed_state_dict[missing_key] = v
                            print(f"Mapped shared embedding: {k} -> {missing_key}")
                            break
                elif 'decoder.embed_tokens.weight' in missing_key:
                    encoder_key = missing_key.replace('decoder', 'encoder')
                    if encoder_key in fixed_state_dict:
                        fixed_state_dict[missing_key] = fixed_state_dict[encoder_key]
                        print(f"Shared decoder embeddings with encoder")
            
            # lm_head shares with decoder embeddings
            elif missing_key == 'lm_head.weight':
                decoder_embed_key = 'model.decoder.embed_tokens.weight'
                if decoder_embed_key in fixed_state_dict:
                    fixed_state_dict[missing_key] = fixed_state_dict[decoder_embed_key]
                    print(f"Shared lm_head with decoder embeddings")
    
    print("--- End fixing state dict ---\n")
    return fixed_state_dict

def test_model(model, tokenizer, device, test_sentences, label="UNTRAINED"):
    """Test model translation"""
    print(f"\n--- Testing {label} model ---")
    model.eval()

    for text in test_sentences:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
            
            # More diverse generation settings for testing
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=5,
                do_sample=False,  # Keep deterministic for testing
            )
        output = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        print(f"XH: {text}")
        print(f"EN: {output}")
        print("---")
    print(f"--- End of {label} model test ---\n")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune NLLB model with evaluation.")
    parser.add_argument('--base_model_path', type=str, required=True, help="Path to model to fine-tune")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to directory containing training and validation data")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save model")
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model manually from: {args.base_model_path}...")
    config = AutoConfig.from_pretrained(args.base_model_path)
    model = AutoModelForSeq2SeqLM.from_config(config)
    
    safetensors_path = os.path.join(args.base_model_path, "model.safetensors")
    state_dict = load_file(safetensors_path, device="cpu")
    
    fixed_state_dict = fix_model_state_dict(state_dict, model)
    
    result = model.load_state_dict(fixed_state_dict, strict=False)
    print(f"\nLoad result: Missing keys: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
    
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, local_files_only=True)
    
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded and moved to GPU.")
    
    # Set generation config
    model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    model.generation_config.max_length = 128
    model.generation_config.num_beams = 5  # Beam search for better quality
    
    # Set decoder_start_token_id if not set
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    
    # Test the base model
    test_sentences_before = [
        "Molo unjani?",
        "Isigulane sinomkhuhlane kunye nentloko."
    ]
    test_model(model, tokenizer, device, test_sentences_before, label="UNTRAINED")
    
    tokenizer.src_lang = "xho_Latn"
    tokenizer.tgt_lang = "eng_Latn"

    # Load and Tokenize Datasets
    def read_text_file(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    raw_datasets = DatasetDict({
        'train': Dataset.from_dict({
            'xh': read_text_file(os.path.join(args.data_dir, 'train.xh')),
            'en': read_text_file(os.path.join(args.data_dir, 'train.en'))
        }),
        'validation': Dataset.from_dict({
            'xh': read_text_file(os.path.join(args.data_dir, 'dev.xh')), 
            'en': read_text_file(os.path.join(args.data_dir, 'dev.en'))
        })
    })

    def preprocess_function(examples):
        # Set source language
        inputs = examples['xh']
        targets = examples['en']
        
        # Tokenize with forced language tokens
        model_inputs = tokenizer(
            inputs, 
            max_length=128, 
            truncation=True,
            padding=False,  
        )
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=128, 
                truncation=True,
                padding=False,  
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets = raw_datasets.map(
        preprocess_function, 
        batched=True, 
        remove_columns=raw_datasets["train"].column_names
    )
    
    # Custom data collator that doesn't create decoder_inputs_embeds
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model,
        padding=True,
        max_length=128,
        pad_to_multiple_of=8, 
        return_tensors="pt",
    )
    
    # Enhanced Metrics Function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_labels = [[label.strip()] for label in decoded_labels]
        decoded_preds = [pred.strip() for pred in decoded_preds]
        
        # Calculate BLEU
        bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
        
        # Calculate chrF (standard) and chrF++ (word_order=2)
        chrf = sacrebleu.corpus_chrf(decoded_preds, decoded_labels, word_order=0)
        chrfpp = sacrebleu.corpus_chrf(decoded_preds, decoded_labels, word_order=2)

        return {
            "bleu": round(bleu.score, 2),
            "chrf": round(chrf.score, 2),
            "chrf++": round(chrfpp.score, 2),
        }

    # Training Arguments with improvements
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=5,
        # Learning rate scheduling
        lr_scheduler_type="linear",
        # Optimizer settings
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        # Early stopping
        eval_on_start=True,  # Get baseline
        report_to="none",  # Disable wandb/tensorboard
        # Additional fixes
        dataloader_drop_last=False,
        group_by_length=True,  # More efficient batching
    )

    # Use custom trainer
    trainer = M2M100Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    print("\n--- Training finished ---")
    
    # Test the fine-tuned model on in-domain sentences
    test_sentences_after = [
        "Olu luvavanyo lwemodeli elungiswe kakuhle.",
        "Kubuhlungu xa ndichama.",
        "Isifuba sam siziva siqine kakhulu.",
        "Ndineentsuku ezintathu ndinentloko.",
        "Ugqirha wamisela amayeza okubulala iintsholongwane."
    ]
    test_model(model, tokenizer, device, test_sentences_after, label="FINE-TUNED")
    
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(f"Final model saved to {args.output_dir}")
    
    # Print best BLEU achieved
    if trainer.state.best_metric:
        print(f"\nBest BLEU achieved: {trainer.state.best_metric}")

if __name__ == "__main__":
    main()