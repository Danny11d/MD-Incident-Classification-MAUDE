# Medical Device Incident Classification

## Project Overview
This project uses textual data from the FDA MAUDE (Manufacturer and User Facility Device Experience) database to automatically classify medical device reports as incident or non-incident.  
Since the 2024 dataset contains no entries with the `Text Type Code = "E"` (which explicitly marks device-related incidents), we use a keyword-based approach to identify potential incidents in the text.  

The main goal is to demonstrate a simple, interpretable NLP pipeline that can help biomedical engineers and analysts quickly flag reports that may involve device malfunctions or safety events.

---

## Dataset
- Source [FDA MAUDE Database/foitext2024.zip]: https://www.fda.gov/medical-devices/medical-device-reporting-mdr-how-report-medical-device-problems/mdr-data-files#download
- Columns used: `MDR_REPORT_KEY`, `FOI_TEXT`
- Rows: 100,000 (subset for performance in Colab/local environments)
- Incident labeling: Based on keywords such as `"fracture"`, `"malfunction"`, `"leak"`, `"fail"`, `"crack"`, etc.

---

## Pipeline

1. Data Loading and Preprocessing
   - Load the `.txt` data file from the `.zip` archive.
   - Drop rows with missing text.
   - Convert all text to lowercase for uniformity.

2. Incident Labeling
   - Assign a binary label `incident = 1` if any of the predefined keywords are present in `FOI_TEXT`.
   - Otherwise, `incident = 0`.

3. Balancing Dataset
   - Sample non-incident rows to match the number of incident rows.
   - Shuffle the dataset to ensure random distribution.

4. Train/Test Split
   - Use an 80/20 split.
   - `stratify=y` ensures the proportion of incidents/non-incidents is preserved in both train and test sets.

5. Feature Extraction
   - Convert text to **TF-IDF features** using unigrams and bigrams.
   - Remove common English stopwords.
   - Limit to the 7,000 most frequent features for memory efficiency.

6. Model Training
   - Logistic Regression classifier.
   - Use `class_weight={0:1, 1:5}` to give more importance to incidents, reducing false negatives.

7. Evaluation
   - Compute **classification report** (precision, recall, F1-score) to assess performance.
   - Visualize **confusion matrix** with a heatmap.
   <img width="548" height="432" alt="matrix" src="https://github.com/user-attachments/assets/d5b19fef-522a-4d8b-8106-625d62cf9d13" />


---

## Results

**Balanced dataset:** equal number of incident and non-incident reports.
**Classification metrics**: high accuracy, precision, and recall for both classes.
**Confusion matrix**: shows model performance visually.
