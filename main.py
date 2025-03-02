# ---------- SECTION 1: IMPORT LIBRARIES ----------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
import pandas as pd
import os

# Check if sentencepiece is installed
try:
    import sentencepiece
except ImportError:
    print("Error: `sentencepiece` is not installed. Please install it using `pip install sentencepiece`.")
    exit(1)

# Check if sacremoses is installed
try:
    import sacremoses
except ImportError:
    print("Warning: `sacremoses` is recommended for Marian models. Install it using `pip install sacremoses`.")

# Increase timeout for Hugging Face downloads
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 300 seconds (5 minutes)

# ---------- SECTION 2: CUSTOM CONFIGURATION ----------
# Add paths to your train and test datasets
TRAIN_DATA_PATH = "/Users/jugal.maniar/Desktop/train.tsv"
TEST_DATA_PATH = "/Users/jugal.maniar/Desktop/test.tsv"

# ---------- SECTION 3: DATASET LOADING ----------
def load_tsv_dataset(file_path):
    """Load a TSV dataset into a pandas DataFrame"""
    try:
        return pd.read_csv(file_path, sep="\t")  # Use '\t' for TSV files
    except Exception as error:
        print(f"Error loading dataset: {str(error)}")
        return None

# ---------- SECTION 4: DATASET PREPROCESSING ----------
def preprocess_dataset(dataset, tokenizer):
    """Preprocess the dataset for Seq2Seq training"""
    def tokenize_function(examples):
        # Tokenize the input (English) and target (Hindi) sequences
        model_inputs = tokenizer(
            examples["en_query"],  # English queries
            max_length=512,
            truncation=True,
            padding="max_length",
        )

        # Tokenize the target (Hindi) sequences
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["cs_query"],  # Hindi (or Hinglish) queries
                max_length=512,
                truncation=True,
                padding="max_length",
            )

        # Add the labels to the model inputs
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Convert the pandas DataFrame to a Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(dataset)

    # Tokenize the dataset
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# ---------- SECTION 5: FINE-TUNE MODEL ----------
def fine_tune_model(train_data, tokenizer):
    """Fine-tune a Seq2Seq model with the custom dataset"""
    try:
        # Load the model
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

        # Preprocess the dataset
        tokenized_train_data = preprocess_dataset(train_data, tokenizer)

        # Split the dataset into training and evaluation sets
        train_eval_split = tokenized_train_data.train_test_split(test_size=0.1)  # 10% for evaluation
        train_dataset = train_eval_split["train"]
        eval_dataset = train_eval_split["test"]

        # Define training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            num_train_epochs=3,  # Adjust as needed
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
            evaluation_strategy="epoch",  # Evaluate every epoch
            predict_with_generate=True,
        )

        # Initialize the Seq2SeqTrainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # Pass the evaluation dataset
            tokenizer=tokenizer,
        )

        # Train the model
        trainer.train()
        return model, tokenizer
    except Exception as error:
        print(f"Error during fine-tuning: {str(error)}")
        return None, None

# ---------- SECTION 6: TEST MODEL ----------
def test_model(test_data, model, tokenizer):
    """Test the model with the test dataset"""
    try:
        for index, row in test_data.iterrows():
            en_query = row["en_query"]  # English query
            cs_query = row["cs_query"]  # Hindi (or Hinglish) query

            # Generate translation
            inputs = tokenizer(en_query, return_tensors="pt", max_length=512, truncation=True, padding=True)
            inputs = {key: value.to("cpu") for key, value in inputs.items()}  # Move inputs to CPU
            outputs = model.generate(**inputs)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"English Query: {en_query}")
            print(f"Expected Hindi Query: {cs_query}")
            print(f"Generated Hindi Query: {translated_text}")
            print("-" * 50)
    except Exception as error:
        print(f"Error during testing: {str(error)}")

# ---------- SECTION 7: MAIN BLOCK ----------
if __name__ == "__main__":
    # Load datasets
    train_data = load_tsv_dataset(TRAIN_DATA_PATH)
    test_data = load_tsv_dataset(TEST_DATA_PATH)

    if train_data is not None and test_data is not None:
        print("Datasets loaded successfully!")

        # Print column names for debugging
        print("Columns in train data:", train_data.columns.tolist())
        print("Columns in test data:", test_data.columns.tolist())

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

        # Fine-tune the model
        print("Fine-tuning model...")
        model, tokenizer = fine_tune_model(train_data, tokenizer)

        if model is not None and tokenizer is not None:
            # Test the model
            print("Testing model...")
            test_model(test_data, model, tokenizer)
        else:
            print("Fine-tuning failed. Exiting...")
    else:
        print("Error loading datasets. Exiting...")