import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('SPAM text message 20170820 - Data.csv')

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['Message'].values)
targets = data['Category'].values

X_train, X_test, y_train, y_test = train_test_split(counts,targets, test_size=0.2)

classifier = MultinomialNB()
classifier = classifier.fit(X_train,y_train)

# check the accuracy score of the model 
y_pred = classifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# save the model
filename = 'SavedModels\\MultinomialNB'
pickle.dump(classifier, open(filename + '.sav', 'wb'))
pickle.dump(vectorizer, open(filename + 'count_vect', 'wb'))

# test the model with new examples
examples = ["Hi John, how are you?",
            "Get free coupons now!!!",
            "Please contact me about job offer. Best regards.",
            "Get your bonus today, limited time!"]

example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)