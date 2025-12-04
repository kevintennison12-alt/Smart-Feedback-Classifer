import pandas as pd
import numpy as np
import sys
import requests
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from pyairtable import Table, Api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# --- 1. Configuration (CRITICAL for Airtable Fetch) ---
AIRTABLE_BASE_ID = 'appLHQZ8AO3UggeWt'
AIRTABLE_TABLE_NAME = 'Feedbacks'
AIRTABLE_API_KEY = 'patXtftaY4JZM7tyH.de544f6763e2a122f30f27220a794c05699088119f7b31561e1d0d492a343f85'

# --- 2. Define Raw Column Names ---
RATING_COLS = [
    '1. Help from Office (Administration, Accounts, Sections, etc.)', '2. Assistance from Exam Section',
    '3.Activities of the Department', '4.Classrooms and Lab Facilities', '5.Library Facilities',
    '6.Internet and WIFI', '7. Canteen/Refreshment Centers (Quality, Hygiene, Service)',
    '8. Hostel and Mess (Rooms, Corridors, Hygiene, Upkeep)', '9.Cultural Events and Activities at Institute Level',
    '10.Drinking Water Availability', '11.Washrooms and its Cleanliness', '12.Bus & Transport',
    '13.Campus Ambience and Security', '14. Overall Teaching, Learning, and Evaluation Process',
    '15. Overall Teaching, Learning, and Evaluation Process'
]
REMARK_COLS = [
    'Remarks(if you have rated Help from Office category below 3)', 'Remarks(if you have rated Assistance from Exam Section category below 3)',
    'Remarks (if you have rated Activities of the Department category below 3)', 'Remarks (if you have rated Classrooms and Lab Facilities category below 3)',
    'Remarks (if you have rated Library Facilities category below 3)', 'Remarks (if you have rated Internet and WIFI category below 3)',
    'Remarks (if you have rated Canteen/Refreshment Centers category below 3)', 'Remarks (if you have rated Hostel and Mess category below 3)',
    'Remarks (if you have rated Cultural Events and Activities category below 3)', 'Remarks (if you have rated Drinking Water Availability category below 3)',
    'Remarks (if you have rated Washrooms and its Cleanliness category below 3)', 'Remarks (if you have rated Bus & Transport category below 3)',
    'Remarks (if you have rated Campus Ambience and Security category below 3)', 'Remarks (if you have rated Overall Teaching category below 3)',
    '15.Other (Any specific suggestion for improvement)'
]

# --- 3. Core Processing Functions ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def convert_stars_to_int(series):
    if series.dtype == 'object':
        series = pd.Series(series).astype(str).str.count(r'\*').fillna(0)
    return pd.to_numeric(series, errors='coerce').fillna(0).astype('Int64')

def clean_and_tokenize(text):
    if pd.isna(text) or text is None: return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    present_rating_cols = [col for col in RATING_COLS if col in df_clean.columns]
    
    if present_rating_cols:
        df_clean.loc[:, present_rating_cols] = df_clean.loc[:, present_rating_cols].fillna(0)
        for col in present_rating_cols:
            df_clean.loc[:, col] = convert_stars_to_int(df_clean[col])
        df_clean['Average_Satisfaction'] = df_clean[present_rating_cols].mean(axis=1).round(2)
    else:
        df_clean['Average_Satisfaction'] = 0

    present_remark_cols = [col for col in REMARK_COLS if col in df_clean.columns]
    if present_remark_cols:
        df_clean.loc[:, present_remark_cols] = df_clean.loc[:, present_remark_cols].fillna('')
        df_clean['Complaint_Text_Raw'] = df_clean[present_remark_cols].apply(lambda row: ' '.join(row.astype(str).fillna('')), axis=1).str.strip()
    else:
        df_clean['Complaint_Text_Raw'] = ''

    df_clean['Complaint_Text_Clean'] = df_clean['Complaint_Text_Raw'].apply(clean_and_tokenize)
    return df_clean

# --- 4. ML Classification ---
def train_and_classify(df: pd.DataFrame):
    print("\n--- Starting Phase 4: Classification and Prediction ---")
    if 'Department' in df.columns:
        df['Category'] = df['Department'].fillna('Other')
    else:
        df['Category'] = 'Other'

    # Heuristic overrides for training labels
    if 'Complaint_Text_Clean' in df.columns:
        df.loc[df['Complaint_Text_Clean'].str.contains('mess|food|hostel|hygiene', case=False, na=False), 'Category'] = 'Hostel/Canteen'
        df.loc[df['Complaint_Text_Clean'].str.contains('wifi|internet|network', case=False, na=False), 'Category'] = 'IT/Technical'
        df.loc[df['Complaint_Text_Clean'].str.contains('exam|marks|sections|office|admin', case=False, na=False), 'Category'] = 'Administration/Exam'
        df.loc[df['Complaint_Text_Clean'].str.contains('bus|transport|route', case=False, na=False), 'Category'] = 'Transport'
        df.loc[df['Complaint_Text_Clean'].str.contains('classroom|lab|furniture|facilities|infra', case=False, na=False), 'Category'] = 'Facilities/Infrastructure'
    
    if df['Category'].nunique() < 2:
        df['Predicted_Category'] = df['Category']
        return df

    if 'Complaint_Text_Clean' not in df.columns:
        df['Predicted_Category'] = df['Category']
        return df

    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_full = vectorizer.fit_transform(df['Complaint_Text_Clean'].fillna(''))
    y_full = df['Category']

    class_counts = y_full.value_counts()
    single_member_classes = class_counts[class_counts <= 1].index.tolist()
    X_stratified, y_stratified = X_full, y_full

    if single_member_classes:
        rows_to_keep = ~y_full.isin(single_member_classes)
        if rows_to_keep.sum() > 0 and rows_to_keep.shape[0] == X_full.shape[0]:
            X_stratified = X_full[rows_to_keep.values, :]
            y_stratified = y_full[rows_to_keep]

    if y_stratified.empty or y_stratified.nunique() < 2:
        df['Predicted_Category'] = df['Category']
    else:
        try:
             test_size = 0.3 if len(y_stratified) >= 10 else 0.1
             X_train, X_test, y_train, y_test = train_test_split(X_stratified, y_stratified, test_size=test_size, random_state=42, stratify=y_stratified)
             model = LogisticRegression(max_iter=1000, class_weight='balanced')
             model.fit(X_train, y_train)
             df['Predicted_Category'] = model.predict(X_full)
             print(f"-> Model trained. Accuracy: {model.score(X_test, y_test):.2f}")
             joblib.dump(model, 'complaint_model.joblib')
             joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
             print("-> Model and vectorizer saved.")
        except Exception as ml_err:
             df['Predicted_Category'] = df['Category']
    return df

# --- 5. Data Retrieval ---
def fetch_and_load_airtable_data():
    try:
        api = Api(AIRTABLE_API_KEY)
        base = api.base(AIRTABLE_BASE_ID)
        table = base.table(AIRTABLE_TABLE_NAME)
        all_records = table.all()
        if not all_records: return pd.DataFrame()

        df = pd.DataFrame([r['fields'] for r in all_records])
        df['Record ID'] = [r['id'] for r in all_records]

        # --- Ensure Urgency/Status/Sentiment columns exist for ML training data preparation ---
        classification_cols = ['Urgency', 'Status', 'Sentiment']
        for col_name in classification_cols:
            if col_name not in df.columns:
                df[col_name] = 'Neutral/Pending' 
            else:
                df[col_name] = df[col_name].astype(str).str.capitalize().fillna('Neutral/Pending').replace('Nan', 'Neutral/Pending')

        return df
    except Exception as e:
        raise Exception(f"Failed during Airtable fetch: {e}")

# --- 6. Execution ---
if __name__ == "__main__":
    try:
        print("Starting data processing pipeline (Model Generation only)...")
        raw_df = fetch_and_load_airtable_data()
        print(f"Fetched {len(raw_df)} records.")

        if not raw_df.empty:
            print("Preprocessing data (Ratings, Text)...")
            final_df = preprocess_data(raw_df)

            # Train ML Model
            final_df = train_and_classify(final_df)

            print("\n-> Model trained and saved to .joblib files.")

        else:
             print("No records fetched from Airtable. Exiting.")

    except KeyError as e:
        print(f"\n--- FATAL ERROR: Column Mismatch ---")
        print(f"A column name in RATING_COLS or REMARK_COLS does not exactly match the Airtable header: {e}")
    except Exception as e:
        print(f"\n--- FATAL PIPELINE ERROR ---")
        print(f"An unexpected error occurred during execution: {e}")