#!/usr/bin/env python3
# generate_synthetic_conll.py
import random
import argparse
import re

# --- Entity lists (expand these as you like) ---
SYMPTOMS = [
    "shortness of breath", "chest pain", "fever", "cough", "fatigue",
    "abdominal pain", "nausea", "headache", "dizziness", "palpitations",
    "sore throat", "loss of taste", "loss of smell", "back pain"
]

DIAGNOSES = [
    "asthma", "pneumonia", "hypertension", "type 2 diabetes", "COPD",
    "heart failure", "acute bronchitis", "urinary tract infection",
    "gastroenteritis", "migraine", "atrial fibrillation"
]

MEDICATIONS = [
    "albuterol", "azithromycin", "amoxicillin", "metformin", "lisinopril",
    "atorvastatin", "prednisone", "insulin", "warfarin", "aspirin"
]

ALLERGIES = [
    "penicillin", "shellfish", "peanuts", "latex", "sulfa drugs",
    "bee stings", "pollen", "mold"
]

PROCEDURES = [
    "chest x-ray", "echocardiogram", "ECG", "CT abdomen", "sputum culture",
    "urinalysis", "bronchoscopy", "colonoscopy"
]

AGE_PHRASES = [
    "a 45-year-old male", "a 62-year-old female", "a 29-year-old female",
    "a 78-year-old male", "a 34-year-old male"
]

# --- Templates to create natural-sounding notes ---
TEMPLATES = [
    "The patient is {age} who presents with {symptom} for {duration}. Past medical history is significant for {diagnosis}. He takes {medication} as needed.",
    "{age} presents complaining of {symptom} and reports {symptom2}. Clinical history includes {diagnosis} and {diagnosis2}. Current meds: {medication}, {med2}.",
    "Patient reports {symptom} over the last {duration}. Exam and vitals are notable. Impression: {diagnosis}. Plan: start {medication} and order {procedure}.",
    "{age} with history of {diagnosis} now with {symptom}. Allergic to {allergy}. Prescribed {medication}.",
    "Chief complaint: {symptom}. HPI: {symptom2} began {duration} ago. Assessment: {diagnosis}. Recommended {procedure} and continue {medication}."
]

DURATIONS = ["3 days", "one week", "2 months", "1 day", "5 days", "several weeks", "overnight"]
DURABLE_ADJ = ["mild", "moderate", "severe", "intermittent", "progressive"]

# --- Simple tokenizer that keeps punctuation as separate tokens ---
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize_with_offsets(text):
    tokens = []
    for m in TOKEN_RE.finditer(text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens

# Create mapping of entity span (in chars) -> label_type
def find_entities_in_text(text, entities_by_type):
    # returns list of tuples (start, end, label, text)
    spans = []
    lowered = text.lower()
    for label_type, entity_list in entities_by_type.items():
        for ent in entity_list:
            # find all occurrences (case-insensitive)
            start = 0
            ent_l = ent.lower()
            while True:
                idx = lowered.find(ent_l, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(ent_l), label_type, text[idx:idx+len(ent_l)]))
                start = idx + len(ent_l)
    # sort spans by start
    spans = sorted(spans, key=lambda x: (x[0], -x[1]))
    # remove overlapping spans: keep longest at same start
    filtered = []
    last_end = -1
    for s in spans:
        if s[0] >= last_end:
            filtered.append(s)
            last_end = s[1]
    return filtered

def label_tokens(text, entities_by_type):
    tokens = tokenize_with_offsets(text)
    token_labels = []
    spans = find_entities_in_text(text, entities_by_type)
    # For quick lookup, create char->(label,start,end) mapping
    char_label = {}
    for start, end, label, ent_text in spans:
        for i in range(start, end):
            char_label[i] = (start, end, label)
    # assign label per token
    for tok, s, e in tokens:
        # find if any char in [s,e) is labeled
        tok_label = "O"
        for i in range(s, e):
            if i in char_label:
                ent_start, ent_end, label = char_label[i]
                # determine B- or I- by comparing token start to entity start
                if s == ent_start:
                    tok_label = "B-" + label
                else:
                    tok_label = "I-" + label
                break
        token_labels.append((tok, tok_label))
    return token_labels

def random_note():
    tpl = random.choice(TEMPLATES)
    # pick variables
    age = random.choice(AGE_PHRASES)
    symptom = random.choice(SYMPTOMS)
    symptom2 = random.choice([s for s in SYMPTOMS if s != symptom])
    diagnosis = random.choice(DIAGNOSES)
    diagnosis2 = random.choice([d for d in DIAGNOSES if d != diagnosis])
    medication = random.choice(MEDICATIONS)
    med2 = random.choice([m for m in MEDICATIONS if m != medication])
    allergy = random.choice(ALLERGIES)
    procedure = random.choice(PROCEDURES)
    duration = random.choice(DURATIONS)
    note = tpl.format(
        age=age,
        symptom=symptom,
        symptom2=symptom2,
        diagnosis=diagnosis,
        diagnosis2=diagnosis2,
        medication=medication,
        med2=med2,
        allergy=allergy,
        procedure=procedure,
        duration=duration
    )
    # add occasional additional sentence
    if random.random() < 0.35:
        extra = f" Patient reports {random.choice(DURABLE_ADJ)} {random.choice(SYMPTOMS)} that started {random.choice(DURATIONS)}."
        note += extra
    return note

def generate_conll_split(n_notes, train_path, test_path, split=0.8):
    entities_by_type = {
        "SYMPTOM": SYMPTOMS,
        "DIAGNOSIS": DIAGNOSES,
        "MEDICATION": MEDICATIONS,
        "ALLERGY": ALLERGIES,
        "PROCEDURE": PROCEDURES
    }

    # Generate all notes first
    all_notes = []
    for _ in range(n_notes):
        note = random_note()
        token_labels = label_tokens(note, entities_by_type)
        all_notes.append(token_labels)

    # Shuffle notes before splitting
    random.shuffle(all_notes)

    # Split index
    split_idx = int(len(all_notes) * split)

    train_notes = all_notes[:split_idx]
    test_notes = all_notes[split_idx:]

    # Write train notes
    with open(train_path, "w", encoding="utf-8") as f_train:
        for note in train_notes:
            for tok, lbl in note:
                f_train.write(f"{tok}\t{lbl}\n")
            f_train.write("\n")

    # Write test notes
    with open(test_path, "w", encoding="utf-8") as f_test:
        for note in test_notes:
            for tok, lbl in note:
                f_test.write(f"{tok}\t{lbl}\n")
            f_test.write("\n")

    print(f"Wrote {len(train_notes)} notes to {train_path}")
    print(f"Wrote {len(test_notes)} notes to {test_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic clinical notes in CoNLL BIO format with train/test split.")
    parser.add_argument("--n", type=int, default=500, help="Number of notes to generate")
    parser.add_argument("--train_out", type=str, default="synthetic_train.conll", help="Training output file path")
    parser.add_argument("--test_out", type=str, default="synthetic_test.conll", help="Testing output file path")
    parser.add_argument("--split", type=float, default=0.8, help="Train/test split fraction (e.g. 0.8)")
    args = parser.parse_args()

    generate_conll_split(args.n, args.train_out, args.test_out, args.split)

if __name__ == "__main__":
    main()
