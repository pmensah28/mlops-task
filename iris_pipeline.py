import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_dataset():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target
    df['species_names'] = df.apply(lambda x: str(data.target_names[int(x['species'])]), axis=1)

    return df

def train(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['species'], test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def get_accuracy(model, X_test, y_test):
    predictions = model.predict(X_test)
    accurcy = accuracy_score(y_test, predictions)

    return accurcy

if __name__ == "__main__":
    data = load_dataset()
    print(data.head())
    model, X_train, X_test, y_train, y_test = train(data)
    accurcy = get_accuracy(model, X_test, y_test)
    print(f"Accuracy: {accurcy:.2f}")