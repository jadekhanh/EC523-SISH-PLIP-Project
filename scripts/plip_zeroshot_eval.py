import pandas as pd
from sklearn.metrics import accuracy_score

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = 'output_predictions.csv'

# Read data from CSV into a DataFrame
df = pd.read_csv(csv_file_path)

# Create a mapping between the actual labels and predicted labels
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

# Convert labels and predicted labels to lowercase (assuming case-insensitive matching)
df['Label'] = df['Label'].str.lower()
df['Predicted_Label'] = df['Predicted_Label'].str.strip('""').str.lower()

# Add a new column for matching
df['Match'] = df['Label'].map(label_mapping) == df['Predicted_Label']

# Calculate the percentage of matches
match_percentage = (df['Match'].sum() / len(df)) * 100

print(f"Percentage of matches: {match_percentage}%")
