"""
Convert EEG2TEXT pickle files to PyTorch-ready format.
Creates train/val/test splits based on subjects.

Usage (Option 1 - specify pickle files directly):
    python scripts/prepare_pytorch_data.py \
        --pickle_files "path/to/task1-SR-dataset.pickle" "path/to/task2-NR-dataset.pickle" \
        --output_dir "./data/processed"

Usage (Option 2 - specify input directory):
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
    pickle_paths: List[Path],
    output_dir: Path,
):
    """
    Process pickle files and save as PyTorch tensors.
    
    Args:
        pickle_paths: List of paths to pickle files
        output_dir: Output directory for processed data
    
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
    stats = {'total': 0, 'skipped': 0, 'by_file': {}}
    
    # Process each pickle file
    for pickle_path in pickle_paths:
        pickle_path = Path(pickle_path)
        
        if not pickle_path.exists():
            print(f"⚠ Warning: {pickle_path} not found, skipping...")
            continue
        
        data = load_pickle(str(pickle_path))
        file_count = 0
        file_name = pickle_path.name
        
        for subject_data in tqdm(data, desc=f"Processing {file_name}"):
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
                file_count += 1
        
        stats['by_file'][file_name] = file_count
    
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
    print(f"  By file: {stats['by_file']}")


def get_pickle_paths_from_dir(input_dir: Path, tasks: List[str]) -> List[Path]:
    """Get pickle file paths from input directory."""
    pickle_paths = []
    for task in tasks:
        pickle_path = input_dir / task / f"{task}-dataset.pickle"
        if not pickle_path.exists():
            pickle_path = input_dir / task / f"{task}-dataset_spectro.pickle"
        if pickle_path.exists():
            pickle_paths.append(pickle_path)
    return pickle_paths


def main():
    parser = argparse.ArgumentParser(
        description='Convert ZuCo pickle files to PyTorch format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Option 1: Specify pickle files directly
  python scripts/prepare_pytorch_data.py --pickle_files task1.pickle task2.pickle
  
  # Option 2: Specify input directory (auto-discovers pickle files)
  python scripts/prepare_pytorch_data.py --input_dir ./dataset/ZuCo
        """
    )
    parser.add_argument('--pickle_files', type=str, nargs='+',
                        help='Paths to pickle files (can specify multiple)')
    parser.add_argument('--input_dir', type=str,
                        help='Directory containing ZuCo pickle files (alternative to --pickle_files)')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='Output directory for PyTorch files')
    parser.add_argument('--tasks', type=str, nargs='+', 
                        default=['task1-SR', 'task2-NR', 'task3-TSR'],
                        help='Tasks to process (used with --input_dir)')
    args = parser.parse_args()
    
    # Get pickle paths
    pickle_paths = []
    
    if args.pickle_files:
        pickle_paths.extend([Path(p) for p in args.pickle_files])
    
    if args.input_dir:
        pickle_paths.extend(get_pickle_paths_from_dir(Path(args.input_dir), args.tasks))
    
    if not pickle_paths:
        parser.error("Must specify either --pickle_files or --input_dir")
    
    print(f"Processing {len(pickle_paths)} pickle file(s):")
    for p in pickle_paths:
        print(f"  - {p}")
    
    process_pickle_files(pickle_paths, Path(args.output_dir))
    
    print("\n✓ Data preparation complete!")


if __name__ == '__main__':
    main()

