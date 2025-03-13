import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file (update the filename as needed)
df = pd.read_csv('TORGO_CLEANED.csv')

# Compute total duration per speaker along with gender and control info.
speaker_stats = df.groupby('spk_id').agg({
    'duration': 'sum',
    'gender': 'first',
    'control': 'first'
}).reset_index()

# Create a stratification column combining gender and control.
speaker_stats['strata'] = speaker_stats['gender'].astype(str) + np.where(speaker_stats['control'], '_control', '')

# Define target ratios for splitting (by total speech time per strata).
target_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}

def split_speakers(group, target_ratios):
    """
    Split speakers in a strata group into train/val/test sets
    based on their total duration.
    """
    total_duration = group['duration'].sum()
    targets = {k: total_duration * v for k, v in target_ratios.items()}
    current = {'train': 0, 'val': 0, 'test': 0}
    assignments = {'train': [], 'val': [], 'test': []}
    
    # Sort speakers in descending order of duration.
    group = group.sort_values('duration', ascending=False)
    
    # Greedily assign each speaker to the set that is the most under its target.
    for _, row in group.iterrows():
        # Compute fill ratio for each set (current/target).
        ratios = {k: (current[k] / targets[k] if targets[k] > 0 else float('inf')) for k in current}
        chosen = min(ratios, key=ratios.get)
        assignments[chosen].append(row['spk_id'])
        current[chosen] += row['duration']
    return assignments, current

# Process each strata group separately.
overall_assignments = {'train': [], 'val': [], 'test': []}
for strata, group in speaker_stats.groupby('strata'):
    assignments, _ = split_speakers(group, target_ratios)
    for key in overall_assignments:
        overall_assignments[key].extend(assignments[key])

# Ensure speakers are not mixed across splits.
train_speakers = overall_assignments['train']
val_speakers = overall_assignments['val']
test_speakers = overall_assignments['test']

# Print the speaker IDs in each split.
print("Training Speakers:", train_speakers)
print("Validation Speakers:", val_speakers)
print("Test Speakers:", test_speakers)

# Prepare the data subsets based on speaker IDs.
train_df = df[df['spk_id'].isin(train_speakers)]
val_df = df[df['spk_id'].isin(val_speakers)]
test_df = df[df['spk_id'].isin(test_speakers)]

traind_df_save = train_df.copy()
val_df_save = val_df.copy()
test_df_save = test_df.copy()

traind_df_save.to_csv('speechbrain/datasets/train.csv', index=False)
val_df_save.to_csv('speechbrain/datasets/val.csv', index=False)
test_df_save.to_csv('speechbrain/datasets/test.csv', index=False)

# Print total durations per set (based on speaker-level aggregation)
train_duration = speaker_stats[speaker_stats['spk_id'].isin(train_speakers)]['duration'].sum()
val_duration   = speaker_stats[speaker_stats['spk_id'].isin(val_speakers)]['duration'].sum()
test_duration  = speaker_stats[speaker_stats['spk_id'].isin(test_speakers)]['duration'].sum()

print("Total Duration (Train):", train_duration)
print("Total Duration (Validation):", val_duration)
print("Total Duration (Test):", test_duration)

# --- Plotting the overview ---
# We'll build two summary DataFrames: one for total duration by set & strata, and one for speaker counts.

def get_duration_by_strata(speaker_ids):
    subset = speaker_stats[speaker_stats['spk_id'].isin(speaker_ids)]
    return subset.groupby('strata')['duration'].sum()

def get_count_by_strata(speaker_ids):
    subset = speaker_stats[speaker_stats['spk_id'].isin(speaker_ids)]
    return subset.groupby('strata').size()

train_duration_by_strata = get_duration_by_strata(train_speakers)
val_duration_by_strata   = get_duration_by_strata(val_speakers)
test_duration_by_strata  = get_duration_by_strata(test_speakers)

durations_df = pd.DataFrame({
    'train': train_duration_by_strata,
    'val': val_duration_by_strata,
    'test': test_duration_by_strata
}).fillna(0)

train_counts = get_count_by_strata(train_speakers)
val_counts   = get_count_by_strata(val_speakers)
test_counts  = get_count_by_strata(test_speakers)

counts_df = pd.DataFrame({
    'train': train_counts,
    'val': val_counts,
    'test': test_counts
}).fillna(0)

# Create a figure with two subplots:
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Left: Stacked bar chart for total speech duration by set (with strata breakdown)
durations_df.T.plot(kind='bar', stacked=True, ax=axs[0], colormap='viridis')
axs[0].set_title('Total Speech Duration by Set & Strata')
axs[0].set_xlabel('Set')
axs[0].set_ylabel('Total Duration')

# Right: Grouped bar chart for the number of speakers by set and strata.
counts_df.T.plot(kind='bar', ax=axs[1], colormap='viridis')
axs[1].set_title('Number of Speakers by Set & Strata')
axs[1].set_xlabel('Set')
axs[1].set_ylabel('Speaker Count')

plt.tight_layout()
plt.savefig('split_overview.png')