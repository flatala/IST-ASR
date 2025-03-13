import pandas as pd

df = pd.read_csv('TORGO_FILTERED.csv')
df.reset_index(drop=True, inplace=True)

df = df[~df['transcription'].str.contains(r'\.jpg|\.png', na=False)]
df['transcription'] = df['transcription'].str.replace(r'\[[^\]]*\]', '', regex=True)
df['transcription'] = df['transcription'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
df = df[df['transcription'].str.strip() != '']
df = df[df['path'] != "TORGO/M05/Session2/wav_headMic/0360.wav"]
df = df[df['duration'] >= 25]

df.rename(columns={'transcription': 'wrd', 'path': 'wav', 'speaker_id': 'spk_id'}, inplace=True)

df.to_csv('TORGO_CLEANED.csv', index=False)