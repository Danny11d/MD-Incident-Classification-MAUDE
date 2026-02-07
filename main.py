import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ZIP_FILE = "foitext2024.zip"       # Path to the zip file containing the FDA MAUDE text dataset
TXT_FILE = "foitext2024.txt"       # The text file inside the zip
MAX_ROWS = 100_000                 # Limit the number of rows of the database

# Keywords for labeling incidents i chose since there are no Code="E"/ wich signifies event/particularly device incident
keywords = ["fracture", "leakage", "malfunction", "break", "crack", "fail", "burst", "detachment", "leak"]



rows = []

# Open the zip file and process 50k rows at a time
with zipfile.ZipFile(ZIP_FILE) as z:
    with z.open(TXT_FILE) as f:
        # Read the text file in chunks to avoid memory issues
        chunks = pd.read_csv(
            f, sep="|", encoding="latin-1", usecols=["MDR_REPORT_KEY", "FOI_TEXT"], chunksize=50_000) # Only need report key + text for NLP 
        
        for chunk in chunks:
            # Remove rows without text
            chunk = chunk.dropna(subset=["FOI_TEXT"])
            
            # Convert to lowercase to standardize
            chunk["FOI_TEXT"] = chunk["FOI_TEXT"].str.lower()
            
    
            # 1 = text contains an incident keyword, 0 = otherwise
            chunk["incident"] = chunk["FOI_TEXT"].apply(
                lambda x: 1 if any(word in x for word in keywords) else 0
            )
            
           
            rows.append(chunk)
            
            # Stop reading further once we reach MAX_ROWS
            if sum(len(x) for x in rows) >= MAX_ROWS:
                break

# Concatenate all chunks into a single DataFrame
df = pd.concat(rows).head(MAX_ROWS)

# Check the shape and distribution of incidents
print(df.shape)
df['incident'].value_counts()


# Separate incident and non-incident rows to balance the dataset
df_incident = df[df['incident']==1]
df_no_incident = df[df['incident']==0].sample(n=len(df_incident), random_state=42)

# Create a balanced dataset
df_balanced = pd.concat([df_incident, df_no_incident]).sample(frac=1, random_state=42)  # Shuffle rows

# Verify distribution
df_balanced['incident'].value_counts()


X = df_balanced['FOI_TEXT']
y = df_balanced['incident']

# Stratified split makes sure proportion of incidents in train/test stays the same
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


tfidf = TfidfVectorizer(stop_words='english', max_features=7000, min_df=5, ngram_range=(1,2))

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight={0:1, 1:5})   # Give more weight to incident class to reduce false negatives
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# Print classification metrics: precision, recall, F1-score
print(classification_report(y_test, y_pred))

#save in the results folder
with open("results/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))


# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True, fmt='d', cmap='Blues',
    xticklabels=['No Incident', 'Incident'],
    yticklabels=['No Incident', 'Incident']
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#save in the results folder
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()
