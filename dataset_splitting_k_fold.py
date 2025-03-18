import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Step 1. Load Data and Compute Speaker-Level Stats
# -----------------------------
df = pd.read_csv('TORGO_CLEANED.csv')

# Compute total duration per speaker along with gender and control info.
speaker_stats = df.groupby('spk_id').agg({
    'duration': 'sum',
    'gender': 'first',
    'control': 'first'
}).reset_index()

# Create a stratification column combining gender and control.
speaker_stats['strata'] = speaker_stats['gender'].astype(str) + np.where(speaker_stats['control'], '_control', '')

# -----------------------------
# Step 2. Partition Speakers into 5 Folds (Stratified by "strata")
# -----------------------------
num_folds = 5
# Initialize dictionaries to hold fold assignments and overall fold durations.
folds = {i: [] for i in range(num_folds)}
folds_duration = {i: 0 for i in range(num_folds)}

# Process each strata group separately.
for strata, group in speaker_stats.groupby('strata'):
    # Sort speakers in descending order of duration.
    group = group.sort_values('duration', ascending=False)
    # For this strata, maintain temporary duration sums per fold.
    strata_fold_duration = {i: 0 for i in range(num_folds)}
    for _, row in group.iterrows():
        # Choose the fold with the smallest total duration (in this strata).
        fold_choice = min(strata_fold_duration, key=strata_fold_duration.get)
        folds[fold_choice].append(row['spk_id'])
        # Update the durations.
        strata_fold_duration[fold_choice] += row['duration']
        folds_duration[fold_choice] += row['duration']

# -----------------------------
# Step 3. Create Folders and Save 5-Fold Train/Test CSVs
# -----------------------------
# Create a main output folder.
output_folder = '5_fold_cv'
os.makedirs(output_folder, exist_ok=True)

# Dictionary to store train/test speaker IDs per fold.
fold_assignments = {}

for i in range(num_folds):
    # For each fold, test speakers are those assigned to fold i.
    test_speakers = folds[i]
    # Train speakers are all speakers not in fold i.
    train_speakers = []
    for j in range(num_folds):
        if j != i:
            train_speakers.extend(folds[j])
    fold_assignments[i] = {'train': train_speakers, 'test': test_speakers}
    
    # Create subfolder for this fold.
    fold_folder = os.path.join(output_folder, f'fold_{i+1}')
    os.makedirs(fold_folder, exist_ok=True)
    
    # Subset the main DataFrame for train and test.
    train_df = df[df['spk_id'].isin(train_speakers)]
    test_df = df[df['spk_id'].isin(test_speakers)]
    
    # Save the CSV files.
    train_df.to_csv(os.path.join(fold_folder, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(fold_folder, 'test.csv'), index=False)
    
    # Print summary stats.
    train_duration = speaker_stats[speaker_stats['spk_id'].isin(train_speakers)]['duration'].sum()
    test_duration = speaker_stats[speaker_stats['spk_id'].isin(test_speakers)]['duration'].sum()
    print(f"Fold {i+1}:")
    print("  Train speakers count:", len(train_speakers), " | Total duration:", train_duration)
    print("  Test speakers count:", len(test_speakers), " | Total duration:", test_duration)
    print("-" * 50)

# -----------------------------
# Step 4. Create and Save a Summary Figure for the 5-Fold Splits
# -----------------------------
# We build dictionaries to store breakdowns (by strata) of total duration and speaker count
# for each fold's train and test sets.
duration_summary = {}
count_summary = {}

for i in range(num_folds):
    fold_label = f"Fold {i+1}"
    test_ids = fold_assignments[i]['test']
    train_ids = fold_assignments[i]['train']
    
    # For the train set:
    train_subset = speaker_stats[speaker_stats['spk_id'].isin(train_ids)]
    train_duration_by_strata = train_subset.groupby('strata')['duration'].sum()
    train_count_by_strata = train_subset.groupby('strata').size()
    
    # For the test set:
    test_subset = speaker_stats[speaker_stats['spk_id'].isin(test_ids)]
    test_duration_by_strata = test_subset.groupby('strata')['duration'].sum()
    test_count_by_strata = test_subset.groupby('strata').size()
    
    duration_summary[f"{fold_label} Train"] = train_duration_by_strata
    duration_summary[f"{fold_label} Test"] = test_duration_by_strata
    count_summary[f"{fold_label} Train"] = train_count_by_strata
    count_summary[f"{fold_label} Test"] = test_count_by_strata

# Convert the dictionaries into DataFrames and fill missing values with 0.
durations_df = pd.DataFrame(duration_summary).fillna(0)
counts_df = pd.DataFrame(count_summary).fillna(0)

# Create a figure with two subplots:
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Left subplot: Stacked bar chart for total speech duration (by fold & set with strata breakdown).
durations_df.T.plot(kind='bar', stacked=True, ax=axs[0], colormap='viridis')
axs[0].set_title('Total Speech Duration by Fold & Set (Strata Breakdown)')
axs[0].set_xlabel('Fold and Set')
axs[0].set_ylabel('Total Duration')

# Right subplot: Grouped bar chart for speaker count (by fold & set with strata breakdown).
counts_df.T.plot(kind='bar', ax=axs[1], colormap='viridis')
axs[1].set_title('Speaker Count by Fold & Set (Strata Breakdown)')
axs[1].set_xlabel('Fold and Set')
axs[1].set_ylabel('Speaker Count')

plt.tight_layout()

# Save the figure in the main output folder.
figure_path = os.path.join(output_folder, 'folds_overview.png')
plt.savefig(figure_path)
plt.close()

print("5-fold cross-validation folders and overview figure have been saved in:", output_folder)
