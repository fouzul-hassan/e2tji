"""Inspect spectro pickle file structure."""
import pickle
import numpy as np

# Use the spectro file this time
path = r"c:\MSc Files\MSc Project\E2T-w-VJEPA\e2t-w-jepa-pretraining\dataset\ZuCo\task1-SR\pickle\task1-SR-dataset-spectro.pickle"

output = []

with open(path, 'rb') as f:
    data = pickle.load(f)

output.append(f"Type: {type(data)}")
output.append(f"Keys: {list(data.keys())}")

# Get first subject's data
first_subject = list(data.keys())[0]
subject_sentences = data[first_subject]

output.append(f"\nSubject: {first_subject}")
output.append(f"Subject data type: {type(subject_sentences)}")
output.append(f"Number of sentences: {len(subject_sentences)}")

# Get first sentence
first_sentence = subject_sentences[0]
output.append(f"\nFirst sentence type: {type(first_sentence)}")
output.append(f"First sentence keys: {list(first_sentence.keys())}")

# Check sentence_level_EEG structure
if 'sentence_level_EEG' in first_sentence:
    eeg_data = first_sentence['sentence_level_EEG']
    output.append(f"\nsentence_level_EEG type: {type(eeg_data)}")
    
    if isinstance(eeg_data, dict):
        output.append(f"sentence_level_EEG keys: {list(eeg_data.keys())}")
        for k, v in list(eeg_data.items())[:3]:
            output.append(f"  {k}: type={type(v).__name__}, shape/len={getattr(v, 'shape', len(v) if hasattr(v, '__len__') else 'N/A')}")
    elif isinstance(eeg_data, np.ndarray):
        output.append(f"sentence_level_EEG shape: {eeg_data.shape}")
    elif isinstance(eeg_data, list):
        output.append(f"sentence_level_EEG length: {len(eeg_data)}")
        if len(eeg_data) > 0:
            output.append(f"  First item type: {type(eeg_data[0])}")
else:
    output.append("\nNo 'sentence_level_EEG' key found!")
    output.append(f"Available keys: {list(first_sentence.keys())}")
    
    # Check what else might be EEG data
    for key in first_sentence.keys():
        val = first_sentence[key]
        if isinstance(val, (np.ndarray, list)):
            output.append(f"  {key}: {type(val).__name__}, len/shape={getattr(val, 'shape', len(val) if hasattr(val, '__len__') else 'N/A')}")
        elif isinstance(val, dict):
            output.append(f"  {key}: dict with keys {list(val.keys())[:5]}")
        else:
            output.append(f"  {key}: {type(val).__name__}")

# Write to file
with open('spectro_structure.txt', 'w') as f:
    f.write('\n'.join(output))

print("Done! Check spectro_structure.txt")
