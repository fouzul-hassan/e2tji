"""Inspect pickle file structure - deeper dive."""
import pickle

path = r"c:\MSc Files\MSc Project\E2T-w-VJEPA\e2t-w-jepa-pretraining\dataset\ZuCo\task1-SR\pickle\task1-SR-dataset.pickle"

output = []

with open(path, 'rb') as f:
    data = pickle.load(f)

output.append(f"Type: {type(data)}")
output.append(f"Keys: {list(data.keys())}")

# Get first subject's data
first_subject = list(data.keys())[0]
subject_data = data[first_subject]

output.append(f"\nSubject: {first_subject}")
output.append(f"Subject data type: {type(subject_data)}")
output.append(f"Subject data length: {len(subject_data)}")

# Get first sentence
first_sentence = subject_data[0]
output.append(f"\nFirst sentence type: {type(first_sentence)}")

if isinstance(first_sentence, dict):
    output.append(f"First sentence keys: {list(first_sentence.keys())}")
    
    for key in first_sentence.keys():
        val = first_sentence[key]
        if isinstance(val, dict):
            output.append(f"  {key}: dict with keys {list(val.keys())[:5]}")
        elif isinstance(val, list):
            output.append(f"  {key}: list of {len(val)} items")
        elif isinstance(val, str):
            output.append(f"  {key}: '{val[:50]}...'")
        else:
            val_type = type(val).__name__
            output.append(f"  {key}: {val_type}")
    
    # Check for sentence_level_EEG
    if 'sentence_level_EEG' in first_sentence:
        eeg = first_sentence['sentence_level_EEG']
        output.append(f"\nsentence_level_EEG keys: {list(eeg.keys())}")
        for k in list(eeg.keys())[:3]:
            v = eeg[k]
            output.append(f"  {k}: len={len(v) if hasattr(v, '__len__') else 'N/A'}")
            
    # Check for content
    if 'content' in first_sentence:
        output.append(f"\nContent: {first_sentence['content']}")

# Write to file
with open('pickle_structure.txt', 'w') as f:
    f.write('\n'.join(output))

print("Done!")
