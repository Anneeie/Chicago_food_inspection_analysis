import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data=pd.read_csv('data.csv')
# Convert the 'Results' column to a binary classification problem
data['Results'] = data['Results'].map({'Pass': 0, 'Fail': 1})

# Select features and target variable
X = data.drop(['Results'], axis=1)
y = data['Results']



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.33, random_state=42)

one_hot = OneHotEncoder(sparse=False)
X_train_encoded_cols_array = one_hot.fit_transform(X_train[['Facility Type','Inspection Type']])
X_val_encoded_cols_array = one_hot.transform(X_val[['Facility Type','Inspection Type']])
X_test_encoded_cols_array = one_hot.transform(X_test[['Facility Type','Inspection Type']])


categories = np.concatenate(one_hot.categories_)

X_train_encoded_cols=pd.DataFrame(X_train_encoded_cols_array,columns=categories, index=X_train.index)
X_test_encoded_cols = pd.DataFrame(X_test_encoded_cols_array,columns=categories, index=X_test.index)
X_val_encoded_cols = pd.DataFrame(X_val_encoded_cols_array,columns=categories, index=X_val.index)


X_train=pd.concat([X_train_encoded_cols, X_train], axis=1)
X_test =pd.concat([X_test_encoded_cols, X_test], axis=1)
X_val =pd.concat([X_val_encoded_cols, X_val], axis=1)

X_train=X_train.drop(columns=['Facility Type','Inspection Type'])
X_test =X_test.drop(columns=['Facility Type','Inspection Type'])
X_val =X_val.drop(columns=['Facility Type','Inspection Type'])


# Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Model 2: Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Model 3: Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

# Model 4: Gradient Boosting
gradient_boosting = GradientBoostingClassifier()
gradient_boosting.fit(X_train, y_train)

# Evaluate models on the validation set and select the best model
models = [log_reg, decision_tree, random_forest, gradient_boosting]
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
best_model = None
best_f1_score = 0

# datetime_columns = X.select_dtypes(include=['datetime64']).columns
# if len(datetime_columns)>0:
#     X[datetime_columns]=X[datetime_columns].astype('float64')

for model, name in zip(models, model_names):
    y_val_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_val_pred)
    print(f"Model: {name}, F1-Score: {f1:.4f}")

    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = model

# Test the best model on the test set
y_test_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("Best Model (on Test Set):", type(best_model).__name__)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")




