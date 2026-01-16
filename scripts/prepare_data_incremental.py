"""
Incremental data preparation for large pickle files.

Processes ONE pickle file at a time and saves incrementally.
Run in separate Colab cells for each pickle file.

Usage:
    # Cell 1: Process first pickle file
    !python scripts/prepare_data_incremental.py \
        --pickle_file "path/to/task1-SR-dataset-spectro.pickle" \
        --output_dir "./data/processed" \
        --mode create  # Creates new files

    # Cell 2: Process second pickle file 
    !python scripts/prepare_data_incremental.py \
        --pickle_file "path/to/task2-NR-dataset-spectro.pickle" \
        --output_dir "./data/processed" \
        --mode append  # Appends to existing files

    # Cell 3, 4, etc...
    !python scripts/prepare_data_incremental.py \
        --pickle_file "path/to/task3-TSR-dataset-spectro.pickle" \
        --output_dir "./data/processed" \
        --mode append

    # Final step: Finalize and optionally split into chunks
    !python scripts/prepare_data_incremental.py \
        --output_dir "./data/processed" \
        --mode finalize \
        --num_chunks 5  # Optional: split into 5 chunks
"""

import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import gc

# ============================================================================
# Constants
# ============================================================================

FREQUENCY_BANDS = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
NUM_CHANNELS = 105
NUM_BANDS = 8
MAX_TIME_SAMPLES = 500  # For spectro data

# ZuCo 1.0 Subject splits (12 subjects) - prefix Z
ZUCO1_TRAIN_SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB']  # 8 subjects
ZUCO1_VAL_SUBJECTS = ['ZKH', 'ZKW']    # 2 subjects
ZUCO1_TEST_SUBJECTS = ['ZMG', 'ZPH']   # 2 subjects

# ZuCo 2.0 Subject splits (18 subjects) - prefix Y
# Subjects: YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YMS, YRH, YRK, YRP, YSD, YSL, YTL
ZUCO2_TRAIN_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YMS']  # 12 subjects
ZUCO2_VAL_SUBJECTS = ['YRH', 'YRK', 'YRP']    # 3 subjects  
ZUCO2_TEST_SUBJECTS = ['YSD', 'YSL', 'YTL']   # 3 subjects

# Combined splits for backward compatibility
TRAIN_SUBJECTS = ZUCO1_TRAIN_SUBJECTS + ZUCO2_TRAIN_SUBJECTS
VAL_SUBJECTS = ZUCO1_VAL_SUBJECTS + ZUCO2_VAL_SUBJECTS
TEST_SUBJECTS = ZUCO1_TEST_SUBJECTS + ZUCO2_TEST_SUBJECTS


def load_pickle(path: str):
    """Load pickle file."""
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def normalize_pickle_data(data) -> List[Dict]:
    """Normalize pickle data to standard format."""
    if isinstance(data, dict):
        first_key = list(data.keys())[0]
        first_val = data[first_key]
        
        if isinstance(first_val, list):
            normalized = []
            for subject_id, sentences in data.items():
                if isinstance(sentences, list):
                    normalized.append({
                        'subject': subject_id,
                        'sentence': sentences
                    })
            return normalized
    
    if isinstance(data, list):
        return data
    
    return []


def extract_sentence_eeg(sentence_obj: Dict, max_time_samples: int = 500) -> Optional[np.ndarray]:
    """Extract EEG features from sentence object."""
    # Handle None sentences
    if sentence_obj is None:
        return None
    
    eeg_data = sentence_obj.get('sentence_level_EEG', {})
    
    # Spectro format (rawData)
    if 'rawData' in eeg_data:
        raw = np.array(eeg_data['rawData'])
        
        if raw.ndim == 1:
            return None
        
        if raw.shape[0] != NUM_CHANNELS:
            if raw.shape[1] == NUM_CHANNELS:
                raw = raw.T
            else:
                return None
        
        T = raw.shape[1]
        if T < max_time_samples:
            padded = np.zeros((NUM_CHANNELS, max_time_samples))
            padded[:, :T] = raw
            eeg = padded
        else:
            eeg = raw[:, :max_time_samples]
        
        # Normalize
        eeg_mean = np.mean(eeg, axis=1, keepdims=True)
        eeg_std = np.std(eeg, axis=1, keepdims=True) + 1e-8
        eeg = (eeg - eeg_mean) / eeg_std
        
        return eeg.astype(np.float32)
    
    # Regular format (frequency bands)
    elif 'mean_t1' in eeg_data:
        features = []
        for band in FREQUENCY_BANDS:
            key = 'mean' + band
            data = eeg_data.get(key)
            if data is None:
                return None
            arr = np.array(data)
            if arr.shape != (NUM_CHANNELS,):
                return None
            features.append(arr)
        
        eeg = np.stack(features, axis=1)
        
        # Normalize
        eeg_mean = np.mean(eeg, axis=0, keepdims=True)
        eeg_std = np.std(eeg, axis=0, keepdims=True) + 1e-8
        eeg = (eeg - eeg_mean) / eeg_std
        
        return eeg.astype(np.float32)
    
    return None


def process_single_pickle(pickle_path: Path, output_dir: Path, mode: str = 'append'):
    """Process a single pickle file incrementally."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Subject to split mapping
    subject_to_split = {}
    for s in TRAIN_SUBJECTS:
        subject_to_split[s] = 'train'
    for s in VAL_SUBJECTS:
        subject_to_split[s] = 'val'
    for s in TEST_SUBJECTS:
        subject_to_split[s] = 'test'
    
    # Temporary storage for this file
    splits = {
        'train': {'eeg': [], 'texts': [], 'subjects': []},
        'val': {'eeg': [], 'texts': [], 'subjects': []},
        'test': {'eeg': [], 'texts': [], 'subjects': []}
    }
    
    stats = {'total': 0, 'skipped': 0}
    
    # Load and process
    data = load_pickle(str(pickle_path))
    data = normalize_pickle_data(data)
    print(f"  Normalized to {len(data)} subjects")
    
    for subject_data in tqdm(data, desc=f"Processing {pickle_path.name}"):
        subject_id = subject_data['subject']
        split = subject_to_split.get(subject_id, 'train')
        
        for sentence in subject_data['sentence']:
            # Skip None sentences
            if sentence is None:
                stats['skipped'] += 1
                continue
            
            eeg = extract_sentence_eeg(sentence)
            if eeg is None:
                stats['skipped'] += 1
                continue
            
            text = sentence.get('content', '').strip()
            if len(text) < 3:
                stats['skipped'] += 1
                continue
            
            splits[split]['eeg'].append(eeg)
            splits[split]['texts'].append(text)
            splits[split]['subjects'].append(subject_id)
            stats['total'] += 1
    
    print(f"\nProcessed: {stats['total']} samples, Skipped: {stats['skipped']}")
    
    # Save incrementally to temp files
    for split_name, split_data in splits.items():
        if len(split_data['eeg']) == 0:
            continue
        
        temp_path = output_dir / f"_temp_{split_name}_{pickle_path.stem}.pt"
        eeg_tensor = torch.tensor(np.stack(split_data['eeg']), dtype=torch.float32)
        
        torch.save({
            'eeg': eeg_tensor,
            'texts': split_data['texts'],
            'subjects': split_data['subjects']
        }, temp_path)
        
        print(f"✓ Saved temp: {temp_path.name} ({len(split_data['texts'])} samples)")
    
    # Memory cleanup
    del data, splits
    gc.collect()
    
    return stats


def finalize_data(output_dir: Path, num_chunks: int = 1):
    """Combine all temp files and optionally split into chunks."""
    output_dir = Path(output_dir)
    
    print("\n" + "="*60)
    print("Finalizing data...")
    print("="*60)
    
    for split_name in ['train', 'val', 'test']:
        # Find all temp files for this split
        temp_files = list(output_dir.glob(f"_temp_{split_name}_*.pt"))
        
        if not temp_files:
            print(f"⚠ No temp files found for {split_name} split")
            continue
        
        print(f"\nCombining {len(temp_files)} files for {split_name}...")
        
        all_eeg = []
        all_texts = []
        all_subjects = []
        
        for temp_file in sorted(temp_files):
            data = torch.load(temp_file, weights_only=False)
            all_eeg.append(data['eeg'])
            all_texts.extend(data['texts'])
            all_subjects.extend(data['subjects'])
            print(f"  Loaded: {temp_file.name} ({len(data['texts'])} samples)")
        
        combined_eeg = torch.cat(all_eeg, dim=0)
        total_samples = len(all_texts)
        
        print(f"  Total: {total_samples} samples, EEG shape: {combined_eeg.shape}")
        
        # Save as chunks or single file
        if num_chunks > 1:
            chunk_size = (total_samples + num_chunks - 1) // num_chunks
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, total_samples)
                
                if start_idx >= total_samples:
                    break
                
                chunk_path = output_dir / f"{split_name}_data_chunk{chunk_idx}.pt"
                torch.save({
                    'eeg': combined_eeg[start_idx:end_idx],
                    'texts': all_texts[start_idx:end_idx],
                    'subjects': all_subjects[start_idx:end_idx]
                }, chunk_path)
                
                print(f"  ✓ Saved: {chunk_path.name} ({end_idx - start_idx} samples)")
        else:
            # Single file
            save_path = output_dir / f"{split_name}_data.pt"
            torch.save({
                'eeg': combined_eeg,
                'texts': all_texts,
                'subjects': all_subjects
            }, save_path)
            print(f"  ✓ Saved: {save_path.name} ({total_samples} samples)")
        
        # Clean up temp files
        for temp_file in temp_files:
            temp_file.unlink()
            print(f"  🗑 Deleted: {temp_file.name}")
    
    # Save metadata
    metadata = {
        'num_channels': NUM_CHANNELS,
        'num_chunks': num_chunks,
        'train_subjects': TRAIN_SUBJECTS,
        'val_subjects': VAL_SUBJECTS,
        'test_subjects': TEST_SUBJECTS,
    }
    torch.save(metadata, output_dir / "metadata.pt")
    print(f"\n✓ Saved metadata.pt")
    
    print("\n✓ Finalization complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Incremental data preparation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  create   - Start fresh, process first pickle file
  append   - Add another pickle file to existing temp data
  finalize - Combine all temp files into final output

Examples:
  # Cell 1: Create with first pickle
  python scripts/prepare_data_incremental.py --pickle_file task1.pickle --mode create
  
  # Cell 2-4: Append more pickle files  
  python scripts/prepare_data_incremental.py --pickle_file task2.pickle --mode append
  python scripts/prepare_data_incremental.py --pickle_file task3.pickle --mode append
  python scripts/prepare_data_incremental.py --pickle_file task4.pickle --mode append
  
  # Cell 5: Finalize and optionally chunk
  python scripts/prepare_data_incremental.py --mode finalize --num_chunks 5
        """
    )
    
    parser.add_argument('--pickle_file', type=str,
                        help='Path to single pickle file to process')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='Output directory')
    parser.add_argument('--mode', type=str, default='append',
                        choices=['create', 'append', 'finalize'],
                        help='Processing mode')
    parser.add_argument('--num_chunks', type=int, default=1,
                        help='Number of chunks to split output (for finalize mode)')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.mode == 'finalize':
        finalize_data(output_dir, args.num_chunks)
    else:
        if not args.pickle_file:
            parser.error("--pickle_file is required for create/append mode")
        
        pickle_path = Path(args.pickle_file)
        if not pickle_path.exists():
            print(f"❌ File not found: {pickle_path}")
            return
        
        if args.mode == 'create':
            # Clean up any existing temp files
            for temp_file in output_dir.glob("_temp_*.pt"):
                temp_file.unlink()
                print(f"🗑 Cleaned: {temp_file.name}")
        
        process_single_pickle(pickle_path, output_dir, args.mode)
        print(f"\n✓ Done! Run more files with --mode append, then --mode finalize")


if __name__ == '__main__':
    main()
