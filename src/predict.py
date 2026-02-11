import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

while True:
    text = input("Enter a sentence (or type 'exit'): ")

    if text.lower() == "exit":
        break

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)

    if prediction[0] == 1:
        print("Positive 😊")
    else:
        print("Negative 😡")
