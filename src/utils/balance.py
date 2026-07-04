import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def data_fix(data):
    data['filepath'] = data['filepath'].apply(lambda x: x.replace('\\', '/'))
    data['class'] = data['class'].replace({
    'repression': 'angry', 
    'disgust':'disgust',
    'fear': 'fear',
    'happiness': 'happy',
    'others': 'neutral',
    'sadness': 'sad',
    'surprise': 'surprise'
    })
    data['class'].value_counts()
    
    return data

def balance_to_middle_sample(df, label_column="class",random_state=42,shuffle=True):
    """
    Balance dataset to the median (middle) class size.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    label_column : str
        Name of the target column.

    random_state : int
        Random seed.

    shuffle : bool
        Shuffle the final dataframe.

    Returns
    -------
    balanced_df : pandas.DataFrame
        Balanced dataframe.
    """

    # Class distribution
    class_counts = df[label_column].value_counts().sort_index()

    print("=" * 70)
    print("ORIGINAL CLASS DISTRIBUTION")
    print("=" * 70)
    print(class_counts)

    # Middle Sample Size (Median)
    middle_sample = int(class_counts.median())

    print("\nMiddle Sample Size:", middle_sample)
    print("=" * 70)

    balanced_data = []

    for class_name, count in class_counts.items():

        class_df = df[df[label_column] == class_name]

        if count > middle_sample:
            # Undersampling
            class_df = resample(
                class_df,
                replace=False,
                n_samples=middle_sample,
                random_state=random_state
            )

        elif count < middle_sample:
            # Oversampling
            class_df = resample(
                class_df,
                replace=True,
                n_samples=middle_sample,
                random_state=random_state
            )

        balanced_data.append(class_df)

    balanced_df = pd.concat(balanced_data)

    if shuffle:
        balanced_df = balanced_df.sample(frac=1,random_state=random_state).reset_index(drop=True)

    print("BALANCED CLASS DISTRIBUTION")
    print("=" * 70)
    print(balanced_df[label_column].value_counts().sort_index())

    print("\nBalanced Dataset Shape:", balanced_df.shape)

    return balanced_df

def data_split(df, output_dir: str, TARGET='class'):
    data_fix(df)
    balance_to_middle_sample(df)
    
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=True,
        stratify=y
    )
    
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values

    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    print("="*60)
    print("Train Shape :", train_df.shape)
    print("Test Shape  :", test_df.shape)
    print("="*60)
    
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    

    
    

    
    
        
    
    