import pandas as pd
from sklearn.datasets import load_iris

def load_dataset():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target
    df['species_names'] = df.apply(lambda x: str(data.target_names[int(x['species'])]), axis=1)

    return df

if __name__ == "__main__":
    data = load_dataset()
    print(data.head())