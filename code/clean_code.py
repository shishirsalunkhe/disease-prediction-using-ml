# =====================================================================================================
# 💻 PROJECT: Disease Prediction using Machine Learning
# 🧠 Developer: Shishir Salunkhe 
# 📋 Description:
#     This project predicts a disease based on symptoms selected by the user.
#     It uses three Machine Learning algorithms:
#         1️⃣ Decision Tree
#         2️⃣ Random Forest
#         3️⃣ Naive Bayes
#     A GUI (Graphical User Interface) is created using Tkinter to make it user-friendly.
# =====================================================================================================

# ------------------------------------- IMPORTING LIBRARIES -------------------------------------------
from tkinter import *           # Tkinter → Used to create GUI windows, labels, buttons, dropdowns, etc.
import numpy as np              # NumPy → For numerical operations and converting data into arrays
import pandas as pd             # Pandas → For loading and handling datasets (CSV files)
# from gui_stuff import *        # (Commented) Optional GUI styling file (not required to run the app)

# ------------------------------------- SYMPTOMS LIST -------------------------------------------------
# l1 contains all possible symptoms that the model uses to make predictions.
# Each symptom acts like a "feature" (column) in our training dataset.
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
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria',
'family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding',
'distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum',
'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads',
'scurrying','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails',
'blister','red_sore_around_nose','yellow_crust_ooze']

# ------------------------------------- DISEASE LIST --------------------------------------------------
# disease → list of all diseases (target labels) that the model can predict.
disease = ['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
'Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria',
'Chicken pox','Dengue','Typhoid','hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D',
'Hepatitis E','Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia',
'Dimorphic hemmorhoids(piles)','Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism',
'Hypoglycemia','Osteoarthristis','Arthritis','(vertigo) Paroymsal  Positional Vertigo',
'Acne','Urinary tract infection','Psoriasis','Impetigo']

# ------------------------------------- INITIALIZING ZERO LIST ----------------------------------------
# l2 → temporary input vector filled with zeros (length same as number of symptoms).
# When user selects symptoms, those positions in l2 become 1 (indicating "symptom present").
l2 = []
for x in range(0, len(l1)):
    l2.append(0)

# ------------------------------------- TRAINING DATA PREPARATION -------------------------------------
df = pd.read_csv("Training.csv")    # Read the training dataset

# Replace disease names with numeric values for ML model training
df.replace({'prognosis': {'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,
'Drug Reaction':4,'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,
'Bronchial Asthma':9,'Hypertension ':10,'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,
'Typhoid':18,'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,
'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,'Common Cold':26,'Pneumonia':27,
'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,
'Psoriasis':39,'Impetigo':40}}, inplace=True)

# Split features and target
X = df[l1]              # All symptom columns (input)
y = df[["prognosis"]]   # Disease label (output)
np.ravel(y)             # Convert to 1D array (required by scikit-learn)

# ------------------------------------- TEST DATA PREPARATION -----------------------------------------
tr = pd.read_csv("Testing.csv")     # Load test dataset
tr.replace({'prognosis': {'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,
'Drug Reaction':4,'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,
'Bronchial Asthma':9,'Hypertension ':10,'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,
'Typhoid':18,'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,
'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,'Common Cold':26,'Pneumonia':27,
'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,
'Psoriasis':39,'Impetigo':40}}, inplace=True)

X_test = tr[l1]               # Test features
y_test = tr[["prognosis"]]    # Test target
np.ravel(y_test)

# ------------------------------------- DECISION TREE FUNCTION ----------------------------------------
def DecisionTree():
    from sklearn import tree                       # Import Decision Tree Classifier
    from sklearn.metrics import accuracy_score     # To calculate model accuracy

    clf3 = tree.DecisionTreeClassifier()           # Create Decision Tree object
    clf3 = clf3.fit(X, y)                          # Train model on training data

    # Check model accuracy
    y_pred = clf3.predict(X_test)
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))

    # Get user symptoms from dropdowns
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

    # Convert user symptoms into binary input vector
    for k in range(0, len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted = predict[0]

    # Map numeric prediction to disease name
    if predicted in range(len(disease)):
        t1.delete("1.0", END)
        t1.insert(END, disease[predicted])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")

# ------------------------------------- RANDOM FOREST FUNCTION ----------------------------------------
def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X, np.ravel(y))

    y_pred = clf4.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    for k in range(0, len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted = predict[0]

    if predicted in range(len(disease)):
        t2.delete("1.0", END)
        t2.insert(END, disease[predicted])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")

# ------------------------------------- NAIVE BAYES FUNCTION ------------------------------------------
def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    gnb = GaussianNB()
    gnb = gnb.fit(X, np.ravel(y))

    y_pred = gnb.predict(X_test)
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    for k in range(0, len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted = predict[0]

    if predicted in range(len(disease)):
        t3.delete("1.0", END)
        t3.insert(END, disease[predicted])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")

# ------------------------------------- TKINTER GUI CREATION ------------------------------------------
root = Tk()                          # Create main application window
root.configure(background='white')   # Set background color

# Variables to store user selections
Symptom1 = StringVar(); Symptom1.set(None)
Symptom2 = StringVar(); Symptom2.set(None)
Symptom3 = StringVar(); Symptom3.set(None)
Symptom4 = StringVar(); Symptom4.set(None)
Symptom5 = StringVar(); Symptom5.set(None)
Name = StringVar()

# -------------------- HEADING --------------------
w2 = Label(root, text="Disease Predictor using Machine Learning", fg="black", bg="white")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=0, columnspan=2, padx=100)

w2 = Label(root, text="A Project by Shishir Salunkhe", fg="Black", bg="white")
w2.config(font=("Aharoni", 20))
w2.grid(row=2, column=0, columnspan=2, padx=100)

# -------------------- LABELS --------------------
NameLb = Label(root, text="Name of the Patient", fg="yellow", bg="black")
NameLb.grid(row=6, column=0, pady=15, sticky=W)

S1Lb = Label(root, text="Symptom 1", fg="yellow", bg="black")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="yellow", bg="black")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="yellow", bg="black")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="yellow", bg="black")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="yellow", bg="black")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)

# -------------------- OUTPUT LABELS --------------------
lrLb = Label(root, text="DecisionTree", fg="white", bg="red")
lrLb.grid(row=15, column=0, pady=10, sticky=W)

destreeLb = Label(root, text="RandomForest", fg="white", bg="red")
destreeLb.grid(row=17, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="NaiveBayes", fg="white", bg="red")
ranfLb.grid(row=19, column=0, pady=10, sticky=W)

# -------------------- INPUT FIELDS --------------------
OPTIONS = sorted(l1)   # Sort symptoms alphabetically for dropdowns

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1En = OptionMenu(root, Symptom1, *OPTIONS); S1En.grid(row=7, column=1)
S2En = OptionMenu(root, Symptom2, *OPTIONS); S2En.grid(row=8, column=1)
S3En = OptionMenu(root, Symptom3, *OPTIONS); S3En.grid(row=9, column=1)
S4En = OptionMenu(root, Symptom4, *OPTIONS); S4En.grid(row=10, column=1)
S5En = OptionMenu(root, Symptom5, *OPTIONS); S5En.grid(row=11, column=1)

# -------------------- BUTTONS --------------------
dst = Button(root, text="DecisionTree", command=DecisionTree, bg="green", fg="yellow")
dst.grid(row=8, column=3, padx=10)

rnf = Button(root, text="Randomforest", command=randomforest, bg="green", fg="yellow")
rnf.grid(row=9, column=3, padx=10)

lr = Button(root, text="NaiveBayes", command=NaiveBayes, bg="green", fg="yellow")
lr.grid(row=10, column=3, padx=10)

# -------------------- TEXT OUTPUT BOXES --------------------
t1 = Text(root, height=1, width=40, bg="orange", fg="black")
t1.grid(row=15, column=1, padx=10)

t2 = Text(root, height=1, width=40, bg="orange", fg="black")
t2.grid(row=17, column=1, padx=10)

t3 = Text(root, height=1, width=40, bg="orange", fg="black")
t3.grid(row=19, column=1, padx=10)

# -------------------- RUN THE APP --------------------
root.mainloop()  # Keeps the window running continuously until closed
