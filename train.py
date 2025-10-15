import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification,pipeline
import numpy as np
import evaluate

# 1. Load data from CoNLL-style file
def read_conll(filepath):
    tokens = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        sentence_tokens = []
        sentence_labels = []
        for line in f:
            line = line.strip()
            if not line:
                if sentence_tokens:
                    tokens.append(sentence_tokens)
                    labels.append(sentence_labels)
                    sentence_tokens = []
                    sentence_labels = []
            else:
                splits = line.split()
                token = splits[0]
                label = splits[1] if len(splits) > 1 else "O"
                sentence_tokens.append(token)
                sentence_labels.append(label)
        # last sentence
        if sentence_tokens:
            tokens.append(sentence_tokens)
            labels.append(sentence_labels)
    return {"tokens": tokens, "labels": labels}

train_data = read_conll("train.conll")
test_data = read_conll("test.conll")

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

# 2. Define labels and mappings
label_list = sorted(set(label for labels in train_data["labels"] for label in labels))
label_to_id = {l: i for i, l in enumerate(label_list)}

# 3. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))

# 4. Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=128)
    labels = []
    for i, label_seq in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special token, ignored in loss
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label_seq[word_idx]])
            else:
                # For subwords inside a word, assign I- label if original label is B-
                current_label = label_seq[word_idx]
                if current_label.startswith("B-"):
                    current_label = current_label.replace("B-", "I-")
                label_ids.append(label_to_id.get(current_label, label_to_id["O"]))
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

# 5. Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# 6. Metric for evaluation
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 9. Train
trainer.train()

trainer.save_model("ner_model")
tokenizer.save_pretrained("ner_model")
