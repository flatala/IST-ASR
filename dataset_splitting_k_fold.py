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

# Create a stratification column if you still want to visualize by strata in final plots
speaker_stats['strata'] = speaker_stats['gender'].astype(str) + np.where(speaker_stats['control'], '_control', '')

# -----------------------------
# Step 2. Partition Speakers into 5 Folds (Balanced by total duration)
# -----------------------------
num_folds = 5
folds = {i: [] for i in range(num_folds)}
folds_duration = {i: 0 for i in range(num_folds)}

# Sort all speakers by descending total duration
speaker_stats_sorted = speaker_stats.sort_values('duration', ascending=False)

# Assign each speaker to the fold with the smallest total duration so far
for _, row in speaker_stats_sorted.iterrows():
    fold_choice = min(folds_duration, key=folds_duration.get)
    folds[fold_choice].append(row['spk_id'])
    folds_duration[fold_choice] += row['duration']

# -----------------------------
# Step 3. Create Folders and Save 5-Fold Train/Test CSVs
# -----------------------------
output_folder = '5_fold_cv'
os.makedirs(output_folder, exist_ok=True)

fold_assignments = {}
for i in range(num_folds):
    test_speakers = folds[i]
    train_speakers = []
    for j in range(num_folds):
        if j != i:
            train_speakers.extend(folds[j])
    fold_assignments[i] = {'train': train_speakers, 'test': test_speakers}
    
    # Create subfolder for this fold
    fold_folder = os.path.join(output_folder, f'fold_{i+1}')
    os.makedirs(fold_folder, exist_ok=True)
    
    # Subset the main DataFrame for train and test
    train_df = df[df['spk_id'].isin(train_speakers)]
    test_df = df[df['spk_id'].isin(test_speakers)]
    
    train_df.to_csv(os.path.join(fold_folder, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(fold_folder, 'test.csv'), index=False)
    
    # Print summary stats
    train_duration = speaker_stats[speaker_stats['spk_id'].isin(train_speakers)]['duration'].sum()
    test_duration = speaker_stats[speaker_stats['spk_id'].isin(test_speakers)]['duration'].sum()
    print(f"Fold {i+1}:")
    print("  Train speakers count:", len(train_speakers), " | Total duration:", train_duration)
    print("  Test speakers count:", len(test_speakers), " | Total duration:", test_duration)
    print("-" * 50)

# -----------------------------
# Print Fold-Speaker Assignment Details
# -----------------------------
print("\nDetailed Fold Assignments:")
for i in range(num_folds):
    print(f"Fold {i+1}: {folds[i]}")

# -----------------------------
# Step 4. Create and Save a Summary Figure for the 5-Fold Splits
# -----------------------------
# Build dictionaries to store breakdowns by strata for each fold's train/test sets
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

# Convert to DataFrame and fill missing values
durations_df = pd.DataFrame(duration_summary).fillna(0)
counts_df = pd.DataFrame(count_summary).fillna(0)

fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Left subplot: stacked bar chart for total speech duration
durations_df.T.plot(kind='bar', stacked=True, ax=axs[0], colormap='viridis')
axs[0].set_title('Total Speech Duration by Fold & Set (Strata Breakdown)')
axs[0].set_xlabel('')
axs[0].set_ylabel('Total Duration')

# Right subplot: grouped bar chart for speaker count
counts_df.T.plot(kind='bar', ax=axs[1], colormap='viridis')
axs[1].set_title('Speaker Count by Fold & Set (Strata Breakdown)')
axs[1].set_xlabel('')
axs[1].set_ylabel('Speaker Count')

plt.tight_layout(rect=[0, 0.15, 1, 1])

# Create a text block with fold assignments
table_text = ""
for i in range(num_folds):
    speakers_str = ", ".join(map(str, folds[i]))
    table_text += f"Fold {i+1}: {speakers_str}\n"

plt.figtext(0.5, 0.02, table_text, wrap=True, horizontalalignment='center', fontsize=10)

# Save the figure
figure_path = os.path.join(output_folder, 'folds_overview.png')
plt.savefig(figure_path)
plt.close()

print("\n5-fold cross-validation folders, speaker assignments, and overview figure have been saved in:", output_folder)
