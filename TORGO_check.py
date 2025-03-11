import pandas as pd

df = pd.read_csv('TORGO_CLEANED.csv')
print(df[df['transcription'].str.contains(r'\d', na=False)])