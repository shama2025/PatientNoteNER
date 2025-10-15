from transformers import AutoTokenizer, AutoModelForTokenClassification,pipeline
import huggingface_hub

def extract_entities(tokens, label_ids, label_map):
    entities = []
    current_entity = ""
    current_type = None

    for token, label_id in zip(tokens, label_ids):
        label = label_map[label_id]

        if label == "O":
            if current_entity:
                entities.append((current_entity, current_type))
                current_entity = ""
                current_type = None
            continue

        prefix, entity_type = label.split("-")

        # Start of a new entity
        if prefix == "B":
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = token if not token.startswith("##") else token[2:]
            current_type = entity_type

        # Continuation of the current entity
        elif prefix == "I" and entity_type == current_type:
            if token.startswith("##"):
                current_entity += token[2:]
            else:
                current_entity += " " + token
        else:
            # Unexpected transition — treat as a new entity
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = token if not token.startswith("##") else token[2:]
            current_type = entity_type

    # Add any remaining entity
    if current_entity:
        entities.append((current_entity, current_type))

    return entities

huggingface_hub.login(token="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")

tokenizer = AutoTokenizer.from_pretrained("mshaffer25/NERPatientNotes")
model = AutoModelForTokenClassification.from_pretrained("mshaffer25/NERPatientNotes")

# Create a pipeline for NER (Named Entity Recognition)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Example input text
text = "The patient has asthma and takes albuterol."

# Run inference
entities = ner_pipeline(text)

# Print result
label_ids = []
tokens = []
label_map = {
    0: 'B-ALLERGY',
    1: 'B-DIAGNOSIS',
    2: 'B-MEDICATION',
    3: 'B-PROCEDURE',
    4: 'B-SYMPTOM',
    5: 'I-ALLERGY',
    6: 'I-DIAGNOSIS',
    7: 'I-PROCEDURE',
    8: 'I-SYMPTOM',
    9: 'O'
}
for ent in entities:
    tokens.append(ent['word'])
    label_ids.append(int(ent['entity_group'].split("_")[1]))
    
print(tokens)
print(label_ids)
print(extract_entities(tokens,label_ids,label_map))