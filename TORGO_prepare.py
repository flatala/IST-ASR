import os
import wave
import pandas as pd
from tqdm import tqdm

DATA_FOLDER = "TORGO"
NO_TRANCRIPTIONS_FOUND_LOG = "no_transcriptions_found.log"

missing_transcriptions: dict[str,str] = {} # k: prompt_id, v: path to wav file

df = pd.DataFrame(columns=['ID', 'duration', 'path', 'speaker_id', 'transcription', 'micsetup', 'gender', 'control', 'session' ])

speakers = os.listdir(DATA_FOLDER)
counter_id = 0

# loop over all speaker folders
for speaker in tqdm(speakers, desc="Processing speakers"):
    gender = "F"
    if speaker.startswith("M"):
        gender = "M"

    control = False
    if speaker[1] == "C":
        control = True

    speaker_folder = os.path.join(DATA_FOLDER, speaker)
    sessions = os.listdir(speaker_folder)

    for session in sessions:
        session_folder = os.path.join(speaker_folder, session)
        if os.path.isdir(session_folder) and 'Session' in session_folder:

            print(session_folder)

            session_contents = os.listdir(session_folder)

            # find the wav_arrayMic, wav_headMic, and prompts folders
            prompts_folder = os.path.join(session_folder, "prompts")
            all_prompts: dict[str, str] = {} # Key: prompt_id, Value: transcription
            if os.path.exists(prompts_folder):
                prompt_files = os.listdir(prompts_folder)
                for prompt_file in prompt_files:
                    if '.txt' in prompt_file:  # we expect only one line per text file, but we'll read all lines just in case
                        prompt_id = prompt_file.split('/')[-1].split('.')[0]
                        with open(os.path.join(prompts_folder, prompt_file), 'r') as f:
                            all_prompts[prompt_id] = f.read().strip()


            wav_arrayMic = os.path.join(session_folder, "wav_arrayMic")
            if os.path.exists(wav_arrayMic):
                wav_files = os.listdir(wav_arrayMic)
                for wav_file in wav_files:
                    # check if the wav file has a corresponding prompt
                    prompt_id = wav_file.split('/')[-1].split('.')[0]
                    transcription = all_prompts.get(prompt_id, None)    
                    if transcription is None:
                        # print(f"Warning: no transcription found for {wav_file}")
                        missing_transcriptions[prompt_id] = os.path.join(wav_arrayMic, wav_file)
                    else:
                        with wave.open(os.path.join(wav_arrayMic, wav_file), 'r') as w:
                            duration = w.getnframes() / w.getframerate() * 1000


                        counter_id += 1
                        df = pd.concat([df, pd.DataFrame({
                                                'ID': [counter_id],
                                                'duration': [duration],
                                                'path': [os.path.join(wav_arrayMic, wav_file)],
                                                'speaker_id': [speaker],
                                                'transcription': [transcription],
                                                'micsetup': ['array'],
                                                'gender': [gender],
                                                'control': [control],
                                                'session': [session]
                                                })], ignore_index=True)
                

            wav_headMic = os.path.join(session_folder, "wav_headMic")
            if os.path.exists(wav_headMic):
                wav_files = os.listdir(wav_headMic)
                for wav_file in wav_files:
                    # check if the wav file has a corresponding prompt
                    prompt_id = wav_file.split('/')[-1].split('.')[0]
                    transcription = all_prompts.get(prompt_id, None)    
                    if transcription is None:
                        # print(f"Warning: no transcription found for {wav_file}")
                        missing_transcriptions[prompt_id] = os.path.join(wav_headMic, wav_file)
                    else:
                        with wave.open(os.path.join(wav_headMic, wav_file), 'r') as w:
                            duration = w.getnframes() / w.getframerate() * 1000

                        counter_id += 1
                        df = pd.concat([df, pd.DataFrame({
                                                'ID': [counter_id],
                                                'duration': [duration],
                                                'path': [os.path.join(wav_arrayMic, wav_file)],
                                                'speaker_id': [speaker],
                                                'transcription': [transcription],
                                                'micsetup': ['head'],
                                                'gender': [gender],
                                                'control': [control],
                                                'session': [session]
                                                })], ignore_index=True)

df.to_csv("TORGO_FILTERED.csv", index=False)
df.to_pickle("TORGO_FILTERED.pkl")


with open(NO_TRANCRIPTIONS_FOUND_LOG, 'w') as f:
    for k, v in missing_transcriptions.items():
        f.write(f"{k}: {v}\n")