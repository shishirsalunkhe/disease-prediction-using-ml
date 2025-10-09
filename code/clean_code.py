# =====================================================================================================
# 💻 PROJECT: Disease Prediction using Machine Learning
# 🧠 Developer: Shishir Salunkhe 
# 📋 Description:
#     Predicts a disease based on selected symptoms using Decision Tree, Random Forest, and Naive Bayes.
#     GUI created using Tkinter for user-friendly interaction.
# =====================================================================================================

from tkinter import *                 # Tkinter for GUI
import pandas as pd                    # Pandas for data handling
import numpy as np                     # NumPy for numerical operations
from sklearn.tree import DecisionTreeClassifier      # Decision Tree model
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.naive_bayes import GaussianNB           # Naive Bayes model
from sklearn.metrics import accuracy_score           # Accuracy evaluation

# ------------------------------------- SYMPTOMS LIST -------------------------------------------------
# List of all symptoms used for features in ML models
l1 = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of_urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic_patches','watering_from_eyes','increased_appetite','polyuria',
'family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding',
'distention_of_abdomen','history_of_alcohol_consumption','blood_in_sputum',
'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads',
'scurrying','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails',
'blister','red_sore_around_nose','yellow_crust_ooze']

# ------------------------------------- DISEASE LIST --------------------------------------------------
# List of diseases (target variable for ML models)
disease = ['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
'Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria',
'Chicken pox','Dengue','Typhoid','hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D',
'Hepatitis E','Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia',
'Dimorphic hemmorhoids(piles)','Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism',
'Hypoglycemia','Osteoarthristis','Arthritis','(vertigo) Paroymsal  Positional Vertigo',
'Acne','Urinary tract infection','Psoriasis','Impetigo']

# ------------------------------------- INITIALIZE ZERO LIST ------------------------------------------
# List of zeros for initializing symptoms vector (not used in this version)
l2 = [0]*len(l1)

# ------------------------------------- DATA PREPROCESSING --------------------------------------------
def preprocess_data(train_file, test_file):
    """
    Reads CSV files, cleans data, maps diseases to integers,
    and returns training/testing datasets with existing symptoms.
    """
    # Read CSV files
    df = pd.read_csv(train_file)
    tr = pd.read_csv(test_file)

    # Remove extra spaces from column names and prognosis
    df.columns = df.columns.str.strip()
    tr.columns = tr.columns.str.strip()
    df['prognosis'] = df['prognosis'].str.strip()
    tr['prognosis'] = tr['prognosis'].str.strip()

    # Map disease names to integers for ML models
    disease_map = {d: i for i, d in enumerate(disease)}
    df['prognosis'] = df['prognosis'].map(disease_map)
    tr['prognosis'] = tr['prognosis'].map(disease_map)

    # Drop rows where prognosis is missing
    df.dropna(subset=['prognosis'], inplace=True)
    tr.dropna(subset=['prognosis'], inplace=True)

    # Ensure prognosis is integer type
    df['prognosis'] = df['prognosis'].astype(int)
    tr['prognosis'] = tr['prognosis'].astype(int)

    # Keep only symptoms that exist in dataset
    existing_symptoms = [sym for sym in l1 if sym in df.columns]

    # Separate features (X) and target (y) for training and testing
    X_train = df[existing_symptoms].astype(int)
    y_train = df['prognosis']
    X_test = tr[existing_symptoms].astype(int)
    y_test = tr['prognosis']

    return X_train, y_train, X_test, y_test, existing_symptoms

# Load processed data
X, y, X_test, y_test, available_symptoms = preprocess_data("Training.csv", "Testing.csv")

# ------------------------------------- HELPER FUNCTION ----------------------------------------------
def make_input_vector(psymptoms):
    """
    Converts user-selected symptoms into binary vector for model prediction.
    """
    vec = [0]*len(available_symptoms)  # initialize zero vector
    for symptom in psymptoms:
        if symptom in available_symptoms:
            idx = available_symptoms.index(symptom)
            vec[idx] = 1  # mark 1 for present symptoms
    return [vec]

# ------------------------------------- CLASSIFIER FUNCTIONS ------------------------------------------
def DecisionTree():
    """
    Predict disease using Decision Tree Classifier.
    """
    clf = DecisionTreeClassifier()  # initialize classifier
    clf.fit(X, y)                   # train model
    y_pred = clf.predict(X_test)
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))  # print accuracy

    # Get symptoms from user input
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    predicted = clf.predict(make_input_vector(psymptoms))[0]  # predict disease

    # Display result in Text widget
    t1.delete("1.0", END)
    t1.insert(END, disease[predicted] if predicted in range(len(disease)) else "Not Found")

def RandomForest():
    """
    Predict disease using Random Forest Classifier.
    """
    clf = RandomForestClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    predicted = clf.predict(make_input_vector(psymptoms))[0]

    t2.delete("1.0", END)
    t2.insert(END, disease[predicted] if predicted in range(len(disease)) else "Not Found")

def NaiveBayes():
    """
    Predict disease using Gaussian Naive Bayes Classifier.
    """
    clf = GaussianNB()
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    predicted = clf.predict(make_input_vector(psymptoms))[0]

    t3.delete("1.0", END)
    t3.insert(END, disease[predicted] if predicted in range(len(disease)) else "Not Found")

# ------------------------------------- TKINTER GUI ---------------------------------------------------
root = Tk()
root.configure(background='#f5f5f5')  # light gray background of main window
root.title("Disease Predictor - Pro GUI")

# ------------------------ Variables ------------------------
Symptom1 = StringVar(); Symptom1.set(None)
Symptom2 = StringVar(); Symptom2.set(None)
Symptom3 = StringVar(); Symptom3.set(None)
Symptom4 = StringVar(); Symptom4.set(None)
Symptom5 = StringVar(); Symptom5.set(None)
Name = StringVar()  # patient name input

# ------------------------ Headings -------------------------
Label(root, text="Disease Predictor using Machine Learning", 
      fg="#1a1a1a",       # Dark gray text color
      bg='#f5f5f5',       # Light gray background (same as window)
      font=("Helvetica", 24, "bold")).grid(row=0, column=0, columnspan=2, padx=20, pady=10)

Label(root, text="A Project by Shishir Salunkhe", 
      fg="#555555",       # Medium gray text for subheading
      bg='#f5f5f5',       # Light gray background
      font=("Helvetica", 16)).grid(row=1, column=0, columnspan=2, padx=20, pady=5)

# ------------------------ Labels ---------------------------
label_color = '#1a1a1a'  # Dark gray text for labels
bg_label = '#f5f5f5'     # Light gray background for labels

Label(root, text="Name of the Patient", fg=label_color, bg=bg_label,
      font=("Helvetica", 12, "bold")).grid(row=2, column=0, pady=5, sticky=W)

for i, text in enumerate(["Symptom 1","Symptom 2","Symptom 3","Symptom 4","Symptom 5"], start=3):
    Label(root, text=text, fg=label_color, bg=bg_label,
          font=("Helvetica", 12, "bold")).grid(row=i, column=0, pady=5, sticky=W)

# Prediction Labels
Label(root, text="Decision Tree", fg="white", bg="#FF5733").grid(row=9, column=0, pady=5, sticky=W)  # Red-orange background for Decision Tree label, white text
Label(root, text="Random Forest", fg="white", bg="#FF5733").grid(row=10, column=0, pady=5, sticky=W)   # Same red-orange background for Random Forest label
Label(root, text="Naive Bayes", fg="white", bg="#FF5733").grid(row=11, column=0, pady=5, sticky=W)      # Same red-orange background for Naive Bayes label

# ------------------------ Input Fields ---------------------
OPTIONS = sorted(available_symptoms)
Entry(root, textvariable=Name, width=30).grid(row=2, column=1)  # Patient name entry (default bg/fg)
OptionMenu(root, Symptom1, *OPTIONS).grid(row=3, column=1)
OptionMenu(root, Symptom2, *OPTIONS).grid(row=4, column=1)
OptionMenu(root, Symptom3, *OPTIONS).grid(row=5, column=1)
OptionMenu(root, Symptom4, *OPTIONS).grid(row=6, column=1)
OptionMenu(root, Symptom5, *OPTIONS).grid(row=7, column=1)

# ------------------------ Buttons --------------------------
Button(root, text="Decision Tree", command=DecisionTree, bg="#28a745", fg="white", width=15).grid(row=3, column=3, padx=10)  # Green button, white text
Button(root, text="Random Forest", command=RandomForest, bg="#28a745", fg="white", width=15).grid(row=4, column=3, padx=10)   # Green button, white text
Button(root, text="Naive Bayes", command=NaiveBayes, bg="#28a745", fg="white", width=15).grid(row=5, column=3, padx=10)        # Green button, white text

# ------------------------ Output Boxes ---------------------
t1 = Text(root, height=1, width=40, bg="#ffe066", fg="#1a1a1a")  # Orange-yellow background for Decision Tree result, dark gray text
t1.grid(row=9, column=1, padx=10)
t2 = Text(root, height=1, width=40, bg="#ffe066", fg="#1a1a1a")  # Orange-yellow background for Random Forest result, dark gray text
t2.grid(row=10, column=1, padx=10)
t3 = Text(root, height=1, width=40, bg="#ffe066", fg="#1a1a1a")  # Orange-yellow background for Naive Bayes result, dark gray text
t3.grid(row=11, column=1, padx=10)

# ------------------------ Run App --------------------------
root.mainloop()  # start GUI event loop
