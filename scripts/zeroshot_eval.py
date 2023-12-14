import pandas as pd
from sklearn.metrics import accuracy_score

csv_file_path = 'output_predictions.csv'

df = pd.read_csv(csv_file_path)

label_mapping = {
    'adi': 'adipose tissue',
    'back': 'background',
    'deb': 'debris',
    'lym': 'lymphocytes',
    'muc': 'mucus',
    'mus': 'smooth muscle',
    'norm': 'normal colon mucosa',
    'str': 'cancer-associated stroma',
    'tum': 'colorectal adenocarcinoma epithelium'
}

df['Label'] = df['Label'].str.lower()
df['Predicted_Label'] = df['Predicted_Label'].str.strip('""').str.lower()

df['Match'] = df['Label'].map(label_mapping) == df['Predicted_Label']

# Calculate the percentage of matches
match_percentage = (df['Match'].sum() / len(df)) * 100

print(f"Percentage of matches: {match_percentage}%")
