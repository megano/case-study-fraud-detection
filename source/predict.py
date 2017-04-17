with open('model.pkl') as f:
    model = pickle.load(f)

model.predict(...)
