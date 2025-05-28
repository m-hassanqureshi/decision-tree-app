import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
# Title of the app 
st.title("Decision Tree Classifier on Iris Dataset")
st.subheader("Iris Dataset Classification")

# Load the dataset
df = pd.read_csv('data/iris.csv')
# Display dataset information
st.write("Dataset Overview:") 
st.write(f"📊 Shape of dataset: {df.shape}")
st.write("📈 Basic Statistics:")
st.write(df.describe())

# Add a correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Allow user to select features
st.sidebar.header("Feature Selection")
selected_features = st.sidebar.multiselect(
    "Select features for classification:",
    options=df.drop("species", axis=1).columns,
    default=df.drop("species", axis=1).columns.tolist()
)

# Show sample distribution
st.subheader("Class Distribution")
fig2, ax2 = plt.subplots()
df['species'].value_counts().plot(kind='bar')
plt.title("Distribution of Iris Species")
st.pyplot(fig2)

st.write("Dataset Preview:") 
st.dataframe(df)

# Features and target
X = df.drop("species", axis=1)
y = df["species"]

# Encode class labels to numbers
le = LabelEncoder()
y = le.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy Check
accuracy = round((accuracy_score(y_test, y_pred)) * 100, 2)
st.write(f"🔍 Accuracy: {accuracy}%")

# Visualizations
st.subheader("Feature Importance")
importance = clf.feature_importances_
features = X.columns
sns.barplot(x=importance, y=features)
plt.title("Feature Importance")

# Add confusion matrix visualization
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax3)
plt.title("Confusion Matrix")
st.pyplot(fig3)

# Add classification report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, target_names=le.classes_)
st.text(report)

# Add hyperparameter tuning
st.sidebar.subheader("Model Parameters")
max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 1)

# Retrain model with selected parameters
clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Add learning curves
st.subheader("Learning Curves")
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

fig4, ax4 = plt.subplots()
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc='best')
st.pyplot(fig4)
