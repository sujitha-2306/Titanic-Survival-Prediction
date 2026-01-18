# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset
df = pd.read_csv('train.csv')

# 3. Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 4. Encode Categorical Columns
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# 5. Select Features and Target
X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = df['Survived']

# 6. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Evaluate Model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# 10. Visualize Survivors
sns.countplot(x='Survived', data=df)
plt.show()

# 11. Interactive Prediction
print("\n--- Titanic Survival Prediction ---")
print("Enter passenger details below:")

# Input passenger info
pclass = int(input("Pclass (1=1st, 2=2nd, 3=3rd): "))
sex = input("Sex (male/female): ").lower()
sex = 1 if sex == 'male' else 0
age = float(input("Age: "))
sibsp = int(input("Number of siblings/spouses aboard: "))
parch = int(input("Number of parents/children aboard: "))
fare = float(input("Fare: "))
embarked = input("Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton): ").upper()

# Encode embarked
embarked_dict = {'C': 0, 'Q': 1, 'S': 2}
embarked = embarked_dict.get(embarked, 2)  # default to S if wrong input

# Prepare input for model
new_passenger = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
new_passenger_scaled = scaler.transform(new_passenger)

# Predict
prediction = model.predict(new_passenger_scaled)
print("\nPrediction: ", "Survived ✅" if prediction[0] == 1 else "Did Not Survive ❌")
