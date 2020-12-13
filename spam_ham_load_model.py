import pickle

# load and use the model previously trained and saved

filename = 'SavedModels\\MultinomialNB'
loaded_model = pickle.load(open(filename + '.sav', 'rb'))
count_vect = pickle.load(open(filename + 'count_vect', 'rb'))

examples = ["Hi John, how are you?",
            "Get free coupons now!!!",
            "Please contact me about job offer. Best regards.",
            "Get your bonus today, limited time!",
            "What are u doing John",
            "How are you? Click to see our prize for you special and today only !"]

result = loaded_model.predict(count_vect.transform(examples))
print(result)