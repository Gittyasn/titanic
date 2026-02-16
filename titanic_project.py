
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load Dataset


# Load train.csv
df = pd.read_csv('train.csv')

# Print first 5 rows
print("First 5 rows of the dataset:")
print(df.head())
print("\n" + "="*50 + "\n")

# Print dataset info
print("Dataset Info:")
print(df.info())
print("\n" + "="*50 + "\n")

# Print missing values count
print("Missing Values Count:")
print(df.isnull().sum())
print("\n" + "="*50 + "\n")


# Data Cleaning & Preparation


# Handle missing values properly
# Age -> fill with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Embarked -> fill with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin column
df.drop(columns=['Cabin'], inplace=True)

# Drop PassengerId, Name, Ticket
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

print("Data Cleaning Completed.")
print("\n" + "="*50 + "\n")


# Exploratory Data Analysis (EDA)


sns.set_style("whitegrid")

# 1. Survival count plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df, palette='pastel')
plt.title('Survival Count')
plt.show()
print("Insight: The count plot shows that more people died than survived.")

# 2. Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=df, palette='coolwarm')
plt.title('Survival by Gender')
plt.show()
print("Insight: Females had a much higher survival rate compared to males.")

# 3. Survival by Passenger Class (Pclass)
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Pclass', data=df, palette='viridis')
plt.title('Survival by Passenger Class')
plt.show()
print("Insight: 1st class passengers had a higher chance of survival than 3rd class.")

# 4. Age distribution histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], kde=True, bins=30, color='blue')
plt.title('Age Distribution')
plt.show()
print("Insight: The age distribution is slightly right-skewed, with many young adults.")

# 5. Survival vs Age boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Age', data=df, palette='Set2')
plt.title('Survival vs Age')
plt.show()
print("Insight: The age distribution of survivors and non-survivors is quite similar.")

print("\n" + "="*50 + "\n")


# Encode Categorical Variables


# Encode Sex and Embarked using pd.get_dummies()
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

print("Feature Encoding Completed.")
print(df.head())
print("\n" + "="*50 + "\n")


# Define Features and Target


# Target variable: Survived
y = df['Survived']

# Features: all remaining columns
X = df.drop(columns=['Survived'])


#  Train-Test Split


# 80% training, 20% testing, random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")
print("\n" + "="*50 + "\n")

# Build Models


# Train Logistic Regression (main model)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Train Decision Tree (for comparison)
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

print("Models Trained Successfully.")
print("\n" + "="*50 + "\n")

#  Model Evaluation

models = {'Logistic Regression': log_reg, 'Decision Tree': decision_tree}

for name, model in models.items():
    print(f"--- {name} Evaluation ---")
    y_pred = model.predict(X_test)
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    print("-" * 30 + "\n")

# Print Final Conclusion


print("--- Final Conclusion ---")
log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test))
dt_acc = accuracy_score(y_test, decision_tree.predict(X_test))

if log_reg_acc > dt_acc:
    best_model = "Logistic Regression"
    best_acc = log_reg_acc
else:
    best_model = "Decision Tree"
    best_acc = dt_acc

print(f"Which model performed better: {best_model}")
print(f"Model accuracy: {best_acc:.4f}")
print("Interpretation of confusion matrix: True Positives and True Negatives show correct predictions.")
print("Key survival insights: Females and 1st class passengers had higher survival rates. Age did not strongly separate survival.")
