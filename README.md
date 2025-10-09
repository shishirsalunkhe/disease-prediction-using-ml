# Disease Prediction using Machine Learning

## 💻 Project Overview

**Project Name:** Disease Prediction using Machine Learning  
**Developer:** Shishir Salunkhe  
**Description:**  
This project predicts a disease based on the symptoms you select. It uses three machine learning models—**Decision Tree, Random Forest, and Naive Bayes**—and shows results in a simple Tkinter GUI.  

You just select up to **5 symptoms**, and the program tells you which disease is most likely.

---

## 🧠 Machine Learning Models

### 1. Decision Tree
- **Why:** Easy to understand and follow  
- **How:** Splits data based on symptoms to find the disease

### 2. Random Forest
- **Why:** More accurate than a single decision tree  
- **How:** Creates many trees and combines results for better predictions

### 3. Naive Bayes
- **Why:** Works well with symptoms data (categorical)  
- **How:** Calculates probability of each disease based on selected symptoms

---

## 🗂️ Dataset

- **Training Data:** `Training.csv`  
- **Testing Data:** `Testing.csv`  
- **Features:** Symptoms (e.g., `back_pain`, `fever`, `cough`)  
- **Target:** Disease name  

**Data Steps:**  
1. Remove extra spaces from columns and disease names  
2. Convert disease names to numbers for ML  
3. Remove rows with missing disease names  
4. Only use symptoms that exist in the dataset  

---

## 🎛️ GUI (Tkinter)

- **Input:** Patient name + select 5 symptoms  
- **Buttons:** Predict using Decision Tree, Random Forest, or Naive Bayes  
- **Output:** Shows the predicted disease  

**Colors Used:**  
- **Background:** Light yellow  
- **Labels:** Black background with yellow text (inputs)  
- **Buttons:** Green background with yellow text  
- **Prediction Output:** Orange background with black text  

**GUI Screenshot:**  
![GUI Output](Output/Screenshot_2025-10-09_122626.png)

---

## ⚡ How to Use

1. Install required packages:
```bash
pip install pandas numpy scikit-learn
Keep Training.csv, Testing.csv, and the Output/ folder in the same directory as clean_code.py.
```
Run the Python script:

bash```
Copy code
python clean_code.py
Enter patient name and select symptoms.

Click a button to see which disease is predicted.

📈 Accuracy
The program prints accuracy for each model on the test data in the console.
Random Forest usually gives the best results.

📝 Key Points
Algorithms:

Decision Tree – easy to understand

Random Forest – more accurate

Naive Bayes – handles symptoms data well

Data Handling: Remove empty disease labels, only use existing symptoms

Prediction: Symptoms are converted to numbers, models predict the disease

GUI: Tkinter is used for simple input and output

Performance: Accuracy is checked using a separate testing dataset

📁 File Structure
bash
Copy code
Disease-Prediction-ML/
│
├─ clean_code.py                 # Main program with GUI
├─ Training.csv                  # Training data
├─ Testing.csv                   # Testing data
├─ Output/
│   └─ Screenshot_2025-10-09_122626.png   # GUI Screenshot
└─ README.md                     # This file
👨‍💻 Developer
Shishir Salunkhe – Original creator
