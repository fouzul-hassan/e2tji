"""
EDA Script for Processed ZuCo Data (.pt files)
Works with already-processed data - no null sentence issues!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================
print("="*60)
print("LOADING PROCESSED DATA")
print("="*60)

data_dir = Path('./data/processed')

# Load all splits
train_data = torch.load(data_dir / 'train_data.pt', weights_only=False)
val_data = torch.load(data_dir / 'val_data.pt', weights_only=False)
test_data = torch.load(data_dir / 'test_data.pt', weights_only=False)

print(f"\n✓ Train: {train_data['eeg'].shape[0]:,} samples")
print(f"✓ Val: {val_data['eeg'].shape[0]:,} samples")
print(f"✓ Test: {test_data['eeg'].shape[0]:,} samples")
print(f"✓ Total: {train_data['eeg'].shape[0] + val_data['eeg'].shape[0] + test_data['eeg'].shape[0]:,} samples")
print(f"✓ EEG shape: {train_data['eeg'].shape[1:]} (channels, time)")

# ============================================================
# 2. DATASET OVERVIEW
# ============================================================
print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

# Subject analysis
all_subjects = train_data['subjects'] + val_data['subjects'] + test_data['subjects']
subject_counts = Counter(all_subjects)
print(f"\nTotal unique subjects: {len(subject_counts)}")
print(f"Subjects: {sorted(subject_counts.keys())}")

# Text analysis
all_texts = train_data['texts'] + val_data['texts'] + test_data['texts']
word_counts = [len(t.split()) for t in all_texts]
char_counts = [len(t) for t in all_texts]

print(f"\nText Statistics:")
print(f"  Mean words/sentence: {np.mean(word_counts):.1f}")
print(f"  Std words: {np.std(word_counts):.1f}")
print(f"  Min words: {min(word_counts)}")
print(f"  Max words: {max(word_counts)}")

# ============================================================
# 3. VISUALIZATIONS
# ============================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig = plt.figure(figsize=(16, 14))

# --- 3.1 Split Distribution ---
ax1 = fig.add_subplot(3, 3, 1)
splits = ['Train', 'Val', 'Test']
sizes = [train_data['eeg'].shape[0], val_data['eeg'].shape[0], test_data['eeg'].shape[0]]
colors = ['#2ecc71', '#3498db', '#e74c3c']
ax1.pie(sizes, labels=splits, autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Dataset Split Distribution', fontweight='bold')

# --- 3.2 Samples per Subject ---
ax2 = fig.add_subplot(3, 3, 2)
subjects_sorted = sorted(subject_counts.keys())
counts = [subject_counts[s] for s in subjects_sorted]
bars = ax2.bar(range(len(subjects_sorted)), counts, color='steelblue', alpha=0.8)
ax2.set_xticks(range(len(subjects_sorted)))
ax2.set_xticklabels(subjects_sorted, rotation=45, ha='right', fontsize=8)
ax2.set_xlabel('Subject ID')
ax2.set_ylabel('Number of Sentences')
ax2.set_title('Samples per Subject', fontweight='bold')

# --- 3.3 Word Count Distribution ---
ax3 = fig.add_subplot(3, 3, 3)
ax3.hist(word_counts, bins=30, color='coral', edgecolor='black', alpha=0.7)
ax3.axvline(np.mean(word_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(word_counts):.1f}')
ax3.axvline(np.median(word_counts), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(word_counts):.1f}')
ax3.set_xlabel('Words per Sentence')
ax3.set_ylabel('Frequency')
ax3.set_title('Sentence Length Distribution', fontweight='bold')
ax3.legend()

# --- 3.4 Sample EEG Signal ---
ax4 = fig.add_subplot(3, 3, 4)
sample_eeg = train_data['eeg'][0].numpy()  # First sample
# Plot a few channels
for ch in [0, 25, 50, 75, 100]:
    ax4.plot(sample_eeg[ch, :200], alpha=0.7, linewidth=0.8, label=f'Ch {ch}')
ax4.set_xlabel('Time (samples)')
ax4.set_ylabel('Amplitude')
ax4.set_title('Sample EEG Signal (Multiple Channels)', fontweight='bold')
ax4.legend(loc='upper right', fontsize=8)

# --- 3.5 EEG Channel Mean/Std ---
ax5 = fig.add_subplot(3, 3, 5)
# Compute across all training samples
eeg_mean = train_data['eeg'].mean(dim=(0, 2)).numpy()  # Mean across samples and time
eeg_std = train_data['eeg'].std(dim=(0, 2)).numpy()
ax5.fill_between(range(105), eeg_mean - eeg_std, eeg_mean + eeg_std, alpha=0.3, color='blue')
ax5.plot(eeg_mean, color='blue', linewidth=1)
ax5.set_xlabel('Channel')
ax5.set_ylabel('Mean ± Std')
ax5.set_title('EEG Channel Statistics', fontweight='bold')

# --- 3.6 Brain Region Power ---
ax6 = fig.add_subplot(3, 3, 6)
BRAIN_REGIONS = {
    'Prefrontal L': list(range(0, 11)),
    'Prefrontal R': list(range(11, 22)),
    'Frontal L': list(range(22, 33)),
    'Frontal R': list(range(33, 44)),
    'Central L': list(range(44, 55)),
    'Central R': list(range(55, 66)),
    'Temporal L': list(range(66, 77)),
    'Temporal R': list(range(77, 88)),
    'Parietal-Occ L': list(range(88, 97)),
    'Parietal-Occ R': list(range(97, 105)),
}
region_powers = []
for name, channels in BRAIN_REGIONS.items():
    power = train_data['eeg'][:, channels, :].abs().mean().item()
    region_powers.append(power)
    
ax6.barh(list(BRAIN_REGIONS.keys()), region_powers, color=plt.cm.viridis(np.linspace(0.2, 0.8, 10)))
ax6.set_xlabel('Average Power')
ax6.set_title('Power by Brain Region', fontweight='bold')

# --- 3.7 EEG Heatmap ---
ax7 = fig.add_subplot(3, 3, 7)
sample_eeg = train_data['eeg'][42].numpy()  # Another sample
im = ax7.imshow(sample_eeg[:, :200], aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
ax7.set_xlabel('Time (samples)')
ax7.set_ylabel('Channel')
ax7.set_title('EEG Heatmap (Sample)', fontweight='bold')
plt.colorbar(im, ax=ax7, label='Amplitude')

# --- 3.8 Text Length vs EEG Power ---
ax8 = fig.add_subplot(3, 3, 8)
# Sample subset for scatter
n_samples = min(500, len(train_data['texts']))
sample_word_counts = [len(train_data['texts'][i].split()) for i in range(n_samples)]
sample_powers = [train_data['eeg'][i].abs().mean().item() for i in range(n_samples)]
ax8.scatter(sample_word_counts, sample_powers, alpha=0.5, c='purple', s=20)
ax8.set_xlabel('Words in Sentence')
ax8.set_ylabel('EEG Power')
ax8.set_title('Text Length vs EEG Power', fontweight='bold')

# --- 3.9 Sample Texts ---
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')
ax9.set_title('Sample Sentences', fontweight='bold')
sample_texts = "\n\n".join([f"[{i+1}] {train_data['texts'][i][:60]}..." for i in range(5)])
ax9.text(0.1, 0.9, sample_texts, transform=ax9.transAxes, fontsize=9, 
         verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('eda_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✓ Saved visualization to eda_visualization.png")

# ============================================================
# 4. DETAILED STATISTICS
# ============================================================
print("\n" + "="*60)
print("DETAILED STATISTICS")
print("="*60)

# EEG statistics
eeg_all = torch.cat([train_data['eeg'], val_data['eeg'], test_data['eeg']], dim=0)
print(f"\nEEG Data Statistics:")
print(f"  Shape: {eeg_all.shape}")
print(f"  Mean: {eeg_all.mean():.4f}")
print(f"  Std: {eeg_all.std():.4f}")
print(f"  Min: {eeg_all.min():.4f}")
print(f"  Max: {eeg_all.max():.4f}")

# Per-split subject distribution
print("\nSubject Distribution per Split:")
print(f"  Train subjects: {len(set(train_data['subjects']))} - {sorted(set(train_data['subjects']))}")
print(f"  Val subjects: {len(set(val_data['subjects']))} - {sorted(set(val_data['subjects']))}")
print(f"  Test subjects: {len(set(test_data['subjects']))} - {sorted(set(test_data['subjects']))}")

print("\n" + "="*60)
print("✅ EDA COMPLETE!")
print("="*60)
