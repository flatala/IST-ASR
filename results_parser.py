import pandas as pd
import re
import argparse
from collections import defaultdict

"""
This script calculates the Word Error Rate (WER) per speaker from multiple evaluation files.

To run this script, use the following command:
python results_parser.py <eval_files> <results_folder>

Example:
python results_parser.py /path/to/eval1.txt /path/to/eval2.txt /path/to/results_folder
(python results_parser.py /scratch/asasin/IST-ASR/speechbrain/transformer/results_lm_60_2_epochs/conformer_small/7775/wer_test.txt 
/scratch/asasin/IST-ASR/speechbrain/transformer/results_lm_50_30_epochs/conformer_small/7775/wer_val.txt 
/scratch/asasin/IST-ASR/speechbrain/transformer/results/)

first file will get the fold id 1 second file will get the fold id 2 etc..

the output folder will look sth like this:
Speaker,WER (%),Errors,Total Words,Fold
MC03,99.83309489747259,4187,4194,1
M01,99.78448275862068,1852,1856,1
FC01,99.70930232558139,686,688,1
F01,99.65034965034964,570,572,1
MC02,228.44196584175668,6554,2869,2
M02,193.64495798319328,3687,1904,2
F04,185.98130841121497,3184,1712,2
FC03,113.53833865814697,5686,5008,2

No matter how many folds you have, the script will calculate the WER per speaker for all folds and save the results in a single CSV file

Arguments:
<eval_files> : Paths to the evaluation files (one per fold)
<results_folder> : Folder to save the results

The script expects the following dataset CSV files to be present:
- /scratch/asasin/IST-ASR/speechbrain/datasets/test.csv
- /scratch/asasin/IST-ASR/speechbrain/datasets/train.csv
- /scratch/asasin/IST-ASR/speechbrain/datasets/val.csv

Each dataset CSV file should contain columns 'ID' and 'spk_id' where:
- 'ID' is the unique identifier for each utterance
- 'spk_id' is the speaker identifier
"""

dataset_paths = [
    "/scratch/asasin/IST-ASR/speechbrain/datasets/test.csv",
    "/scratch/asasin/IST-ASR/speechbrain/datasets/train.csv",
    "/scratch/asasin/IST-ASR/speechbrain/datasets/val.csv",
]

# command line arguments
parser = argparse.ArgumentParser(description="Calculate WER per speaker from multiple evaluation files.")
parser.add_argument("eval_files", type=str, nargs='+', help="Paths to the evaluation files (one per fold)")
parser.add_argument("results_folder", type=str, help="Folder to save the results")
args = parser.parse_args()

# speaker information from dataset CSV files
utterance_to_speaker = {}
for path in dataset_paths:
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        utterance_to_speaker[row['ID']] = row['spk_id']

# process each evaluation file and store results in a single DataFrame
all_results = []
for i, eval_file in enumerate(args.eval_files, start=1):
    wer_data = defaultdict(lambda: {'errors': 0, 'words': 0})
    
    with open(eval_file, 'r') as f:
        for line in f:
            match = re.match(r"(\d+), %WER (\d+\.\d+) \[ (\d+) / (\d+),", line)
            if match:
                utterance_id = int(match.group(1))
                errors = int(match.group(3))
                total_words = int(match.group(4))
                
                speaker = utterance_to_speaker.get(utterance_id, "Unknown")
                wer_data[speaker]['errors'] += errors
                wer_data[speaker]['words'] += total_words

    # WER per speaker and add fold number
    for speaker, data in wer_data.items():
        wer = (data['errors'] / data['words']) * 100 if data['words'] > 0 else 0
        all_results.append((speaker, wer, data['errors'], data['words'], i))

# convert all results to a single CSV
wer_df = pd.DataFrame(all_results, columns=["Speaker", "WER (%)", "Errors", "Total Words", "Fold"])
wer_df.sort_values(by=["Fold", "WER (%)"], ascending=[True, False], inplace=True)
output_file = f"{args.results_folder}/wer_per_speaker_all_folds.csv"
wer_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")