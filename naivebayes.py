#assgn 2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Read dataset
msg = pd.read_csv('document.csv', names=['message', 'label'])

print("Total Instances of Dataset:", msg.shape[0])

# Remove missing values
msg = msg.dropna()

# Convert labels into numbers
msg['labelnum'] = msg['label'].map({'pos': 1, 'neg': 0})

# Remove rows with invalid labels
msg = msg.dropna(subset=['labelnum'])

# Separate input and output
X = msg['message']
y = msg['labelnum']

# Split dataset
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Convert text into numerical features
count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)

# Display word frequency dataframe
df = pd.DataFrame(
    Xtrain_dm.toarray(),
    columns=count_v.get_feature_names_out()
)

print("\nWord Frequency Matrix:")
print(df)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)

# Predict output
pred = clf.predict(Xtest_dm)

# Evaluate model
print('\nAccuracy Metrics:')
print('Accuracy:', accuracy_score(ytest, pred))
print('Recall:', recall_score(ytest, pred, zero_division=0))
print('Precision:', precision_score(ytest, pred, zero_division=0))
print('Confusion Matrix:\n', confusion_matrix(ytest, pred))
