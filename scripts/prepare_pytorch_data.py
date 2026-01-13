"""
Convert EEG2TEXT pickle files to PyTorch-ready format.
Creates train/val/test splits based on subjects.

Usage:
    python scripts/prepare_pytorch_data.py \
        --input_dir "C:/MSc Files/MSc Project/E2T-w-VJEPA/e2t-w-jepa-pretraining/dataset/ZuCo" \
        --output_dir "./data/processed"
"""

import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# ============================================================================
# Constants
# ============================================================================

FREQUENCY_BANDS = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
NUM_CHANNELS = 105
NUM_BANDS = 8

# Subject splits for ZuCo v1.0 (12 subjects)
TRAIN_SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZKB', 'ZKH']
VAL_SUBJECTS = ['ZKW', 'ZMG']
TEST_SUBJECTS = ['ZPH', 'ZRP']

# Brain region channel mapping
BRAIN_REGIONS = {
    'prefrontal_left': list(range(0, 11)),
    'prefrontal_right': list(range(11, 22)),
    'frontal_left': list(range(22, 33)),
    'frontal_right': list(range(33, 44)),
    'central_left': list(range(44, 55)),
    'central_right': list(range(55, 66)),
    'temporal_left': list(range(66, 77)),
    'temporal_right': list(range(77, 88)),
    'parietal_occipital_left': list(range(88, 97)),
    'parietal_occipital_right': list(range(97, 105)),
}


# ============================================================================
# Processing Functions
# ============================================================================

def load_pickle(path: str) -> List[Dict]:
    """Load EEG2TEXT pickle file."""
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_sentence_eeg(sentence_obj: Dict, normalize: bool = True) -> Optional[np.ndarray]:
    """
    Extract sentence-level EEG features.
    
    Returns:
        EEG array of shape (105, 8) or None if invalid
    """
    try:
        features = []
        for band in FREQUENCY_BANDS:
            key = 'mean' + band
            data = sentence_obj['sentence_level_EEG'][key]
            if len(data) != NUM_CHANNELS:
                return None
            features.append(data)
        
        # Stack: (num_bands, num_channels) -> transpose to (num_channels, num_bands)
        eeg = np.stack(features, axis=0).T  # Shape: (105, 8)
        
        if normalize:
            # Per-channel normalization
            mean = eeg.mean(axis=1, keepdims=True)
            std = eeg.std(axis=1, keepdims=True) + 1e-8
            eeg = (eeg - mean) / std
        
        # Check for NaN/Inf
        if np.isnan(eeg).any() or np.isinf(eeg).any():
            return None
            
        return eeg.astype(np.float32)
        
    except Exception as e:
        return None


def process_pickle_files(
    input_dir: Path,
    output_dir: Path,
    tasks: List[str] = ['task1-SR', 'task2-NR', 'task3-TSR']
):
    """
    Process all pickle files and save as PyTorch tensors.
    
    Output files:
        output_dir/
        ├── train_data.pt     # {'eeg': Tensor, 'texts': List, 'subjects': List}
        ├── val_data.pt
        ├── test_data.pt
        └── metadata.pt       # {'brain_regions': dict, 'subjects': dict, ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect samples by split
    splits = {
        'train': {'eeg': [], 'texts': [], 'subjects': []},
        'val': {'eeg': [], 'texts': [], 'subjects': []},
        'test': {'eeg': [], 'texts': [], 'subjects': []}
    }
    
    subject_to_split = {}
    for s in TRAIN_SUBJECTS:
        subject_to_split[s] = 'train'
    for s in VAL_SUBJECTS:
        subject_to_split[s] = 'val'
    for s in TEST_SUBJECTS:
        subject_to_split[s] = 'test'
    
    # Track statistics
    stats = {'total': 0, 'skipped': 0, 'by_task': {}}
    
    # Process each task
    for task in tasks:
        pickle_path = input_dir / task / f"{task}-dataset.pickle"
        
        # Try alternative naming
        if not pickle_path.exists():
            pickle_path = input_dir / task / f"{task}-dataset_spectro.pickle"
        
        if not pickle_path.exists():
            print(f"⚠ Warning: {pickle_path} not found, skipping...")
            continue
        
        data = load_pickle(str(pickle_path))
        task_count = 0
        
        for subject_data in tqdm(data, desc=f"Processing {task}"):
            subject_id = subject_data['subject']
            
            # Determine split
            if subject_id not in subject_to_split:
                print(f"  ⚠ Unknown subject {subject_id}, assigning to train")
                split = 'train'
            else:
                split = subject_to_split[subject_id]
            
            # Process sentences
            for sentence in subject_data['sentence']:
                eeg = extract_sentence_eeg(sentence)
                if eeg is None:
                    stats['skipped'] += 1
                    continue
                
                text = sentence['content'].strip()
                if len(text) < 3:
                    stats['skipped'] += 1
                    continue
                
                splits[split]['eeg'].append(eeg)
                splits[split]['texts'].append(text)
                splits[split]['subjects'].append(subject_id)
                
                stats['total'] += 1
                task_count += 1
        
        stats['by_task'][task] = task_count
    
    # Save each split
    print("\n" + "="*60)
    print("Saving processed data...")
    print("="*60)
    
    for split_name, split_data in splits.items():
        if len(split_data['eeg']) == 0:
            print(f"⚠ Warning: No data for {split_name} split")
            continue
        
        eeg_tensor = torch.tensor(np.stack(split_data['eeg']), dtype=torch.float32)
        
        save_path = output_dir / f"{split_name}_data.pt"
        torch.save({
            'eeg': eeg_tensor,
            'texts': split_data['texts'],
            'subjects': split_data['subjects']
        }, save_path)
        
        print(f"✓ Saved {split_name}: {len(split_data['texts'])} samples -> {save_path}")
        print(f"    EEG shape: {eeg_tensor.shape}")
    
    # Save metadata
    metadata = {
        'brain_regions': BRAIN_REGIONS,
        'frequency_bands': FREQUENCY_BANDS,
        'num_channels': NUM_CHANNELS,
        'num_bands': NUM_BANDS,
        'train_subjects': TRAIN_SUBJECTS,
        'val_subjects': VAL_SUBJECTS,
        'test_subjects': TEST_SUBJECTS,
        'statistics': stats
    }
    torch.save(metadata, output_dir / "metadata.pt")
    
    print(f"\n✓ Saved metadata -> {output_dir / 'metadata.pt'}")
    print(f"\nStatistics:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  By task: {stats['by_task']}")


def main():
    parser = argparse.ArgumentParser(description='Convert ZuCo pickle files to PyTorch format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing ZuCo pickle files')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='Output directory for PyTorch files')
    parser.add_argument('--tasks', type=str, nargs='+', 
                        default=['task1-SR', 'task2-NR', 'task3-TSR'],
                        help='Tasks to process')
    args = parser.parse_args()
    
    process_pickle_files(
        Path(args.input_dir), 
        Path(args.output_dir),
        args.tasks
    )
    
    print("\n✓ Data preparation complete!")


if __name__ == '__main__':
    main()
