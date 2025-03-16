import pandas as pd

df = pd.read_csv('TORGO_FILTERED.csv')
df.reset_index(drop=True, inplace=True)

df = df[~df['transcription'].str.contains(r'\.jpg|\.png', na=False)]
df['transcription'] = df['transcription'].str.replace(r'\[[^\]]*\]', '', regex=True)
df['transcription'] = df['transcription'].str.lower().str.replace(r"[^a-z0-9\s']", '', regex=True)
df = df[df['transcription'].str.strip() != '']
df = df[df['path'] != "TORGO/M05/Session2/wav_headMic/0360.wav"]
df = df[df['duration'] >= 25]
df = df[df['transcription'] != 'xxx']

diddy_files = [
    "TORGO/F01/Session1/wav_headMic/0067.wav",
    "TORGO/F01/Session1/wav_headMic/0068.wav",
    "TORGO/FC01/Session1/wav_arrayMic/0256.wav",
    "TORGO/F03/Session2/wav_arrayMic/0014.wav",
    "TORGO/MC03/Session1/wav_arrayMic/0008.wav",
    "TORGO/MC03/Session2/wav_arrayMic/0250.wav",
    "TORGO/M05/Session1/wav_arrayMic/0007.wav",
    "TORGO/M05/Session1/wav_arrayMic/0047.wav",
    "TORGO/M05/session2/wav_headMic/0093.wav",
    "TORGO/M05/Session2/wav_headMic/0303.wav",
    "TORGO/M05/Session2/wav_headMic/0330.wav",
    "TORGO/M05/Session2/wav_headMic/0331.wav",
    "TORGO/M05/Session2/wav_headMic/0332.wav",
    "TORGO/M05/Session2/wav_headMic/0333.wav",
    "TORGO/M05/Session2/wav_headMic/0334.wav",
    "TORGO/M05/Session2/wav_headMic/0335.wav",
    "TORGO/M05/Session2/wav_headMic/0336.wav",
    "TORGO/M05/Session2/wav_headMic/0337.wav",
    "TORGO/M05/Session2/wav_headMic/0338.wav",
    "TORGO/M05/Session2/wav_headMic/0339.wav",
    "TORGO/M05/Session2/wav_headMic/0340.wav",
    "TORGO/M05/Session2/wav_headMic/0341.wav",
    "TORGO/M05/Session2/wav_headMic/0342.wav",
    "TORGO/M05/Session2/wav_headMic/0343.wav",
    "TORGO/M05/Session2/wav_headMic/0344.wav",
    "TORGO/M05/Session2/wav_headMic/0346.wav",
    "TORGO/M05/Session2/wav_headMic/0347.wav",
    "TORGO/M05/Session2/wav_headMic/0348.wav",
    "TORGO/M05/Session2/wav_headMic/0349.wav",
    "TORGO/M05/Session2/wav_headMic/0350.wav",
    "TORGO/M05/Session2/wav_headMic/0351.wav",
    "TORGO/M05/Session2/wav_headMic/0352.wav",
    "TORGO/M05/Session2/wav_headMic/0353.wav",
    "TORGO/M05/Session2/wav_headMic/0354.wav",
    "TORGO/M05/Session2/wav_headMic/0355.wav",
    "TORGO/M05/Session2/wav_headMic/0356.wav",
    "TORGO/M05/Session2/wav_headMic/0357.wav",
    "TORGO/M05/Session2/wav_headMic/0358.wav",
    "TORGO/M05/Session2/wav_headMic/0359.wav",
    "TORGO/M05/Session2/wav_headMic/0360.wav",
    "TORGO/M05/Session2/wav_headMic/0361.wav",
    "TORGO/F03/Session1/wav_headMic/0118.wav",
    "TORGO/M01/Session2_3/wav_headMic/0180.wav",
    "TORGO/M05/session2/wav_headMic/0345.wav",
    "TORGO/MC04/Session2/wav_arrayMic/0774.wav",
    "TORGO/M04/Session1/wav_arrayMic/0112.wav ",
    "TORGO/MC01/Session2/wav_arrayMic/0201.wav",
    "TORGO/MC04/Session1/wav_arrayMic/0234.wav",
    "TORGO/MC04/Session1/wav_arrayMic/0235.wav",
    "TORGO/MC04/Session1/wav_arrayMic/0427.wav",
]

# drop sus files
df = df.where(~df['path'].isin(diddy_files)).dropna()

df.rename(columns={'transcription': 'wrd', 'path': 'wav', 'speaker_id': 'spk_id'}, inplace=True)

df.to_csv('TORGO_CLEANED.csv', index=False)