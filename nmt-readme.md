# NMT Domain Adaptation via Forward/Back-Translation for isiXhosa Medical Dialogues

This honours project investigates domain adaptation of Neural Machine Translation for the low-resource, specialized context of English-isiXhosa medical dialogues. The core methodology involves fine-tuning pre-trained NLLB models on a synthetic, in-domain dataset created through forward- and back-translation.

## Installation

To create an Anaconda environment with all the necessary dependencies, first ensure you are using a machine with a compatible NVIDIA GPU and CUDA installed.

1. **Create the environment from the YAML file:**
```bash
conda env create -f environment.yml
```

2. **Activate the environment:**
```bash
conda activate medtranslate-fwd-backtrans
```

3. **Install the AfriCOMET library:**

The `unbabel-comet` package must be installed separately via pip after activating the environment.
```bash
pip install unbabel-comet
```

## Generating the Synthetic Data

The synthetic training data is generated through a two-step process. First, a monolingual English corpus is translated into isiXhosa using a powerful baseline NLLB model. Second, the resulting bilingual CSV is split into parallel text files required for training.

### Step 1: Download a Baseline NLLB Model

This process requires a pre-trained NLLB model from the Hugging Face Hub to act as the "teacher" translator. This project utilized two different sizes. You only need to do this once, as the model will be cached locally.

- **NLLB-200-distilled-600M**: [huggingface.co/facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- **NLLB-200-1.3B**: [huggingface.co/facebook/nllb-200-1.3B](https://huggingface.co/facebook/nllb-200-1.3B)

### Step 2: Translate the Monolingual Corpus

Use the `translate_primock.py` script (located in the `data_processing` folder) to generate isiXhosa translations. This script takes an input CSV containing a `text` column (e.g., the Primock57 dataset) and produces an output CSV with a new `isixhosa_translation` column.

**Usage:**
```bash
python data_processing/translate_primock.py \
    --input_file /path/to/primock57.csv \
    --output_file /path/to/translated_primock.csv
```

> **Note:** This script is GPU-intensive. The local path to the baseline NLLB model is hardcoded within the script and may need to be adjusted based on your environment.

### Step 3: Create Parallel Training Files

Use the `create_training_files.py` script (located in the `data_processing` folder) to convert the translated CSV from the previous step into two parallel text files (`.en` and `.xh`).

**Usage:**
```bash
python data_processing/create_training_files.py \
    --input_csv /path/to/translated_primock.csv \
    --output_prefix train_medical_data \
    --output_dir ../finetune_data
```

This will create `../finetune_data/train_medical_data.en` and `../finetune_data/train_medical_data.xh`, which are now ready for fine-tuning.

## Training

This project fine-tunes models for both Forward-Translation (EN→XH) and Back-Translation (XH→EN).

While full hyperparameter sweeps can be run using the provided PBS scripts, a single model can be trained with a specific configuration by calling the training script directly. 

**Example:**
```bash
python train_en_to_xh.py \
    --base_model_path nllb-200/ \
    --data_dir data-bin/ \
    --output_dir finetuned_models/ \
    --learning_rate 5e-6 \
    --batch_size 4 \
    --num_epochs 5 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --label_smoothing 0.1
```

## Evaluation

A robust, multi-metric evaluation harness is used. For BLEU, chrF, and AfriCOMET, a two-step process of generation followed by scoring is required.

### BLEU, chrF and chrF++ Scores

If the hyperparameter sweep was run, then the script `eng_to_xho_eval.sh` / `xho_to_eng_eval.sh` can be run to evaluate all of the models. To evaluate an individual model:

1. **Generate translations of the evaluation set using the model:**
```bash
python generate_translations.py \
    --model_path finetuned_models/model1 \
    --source_file dev.en \
    --output_file model1_predictions.xh \
    --source_lang "eng_Latn" \
    --target_lang "xho_Latn"
```

2. **Generate the BLEU, chrF and chrF++ scores:**
```bash
python verify_scores.py \
    --predictions model1_predictions.xh \
    --references dev.xh
```

### Health Term Error Rate

To calculate the health term error rate run `evaluate_terminology.py` with the following command:

```bash
python evaluate_terminology.py \
    --predictions model1_predictions.xh \
    --references dev.xh \
    --terms_csv medical_terms.csv \
    --direction en_to_xh
```

### AfriCOMET Score

To calculate the AfriCOMET score, first generate translations as shown above, then run the `AfriCOMET_eval.sh` script with the source, prediction, and reference files as arguments.

```bash
./AfriCOMET_eval.sh \
    dev.en \
    model1_predictions.xh \
    dev.xh
```