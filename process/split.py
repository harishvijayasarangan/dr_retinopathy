"""import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(csv_path):
    # Load CSV file and ensure it contains data.
    df = pd.read_csv(csv_path, names=['image_path', 'label'])

    # Debugging: Print dataset info
    print(df.head())
    print(f"Total rows before filtering: {len(df)}")

    # Remove filtering if file paths are not valid
    # df = df[df['image_path'].apply(lambda x: os.path.exists(x))]

    print(f"Total rows after filtering: {len(df)}")
    return df

def main(csv_path, train_csv_path, val_csv_path, test_size=0.2, random_state=42):
    # Load data
    df = load_data(csv_path)

    # Check if dataset is empty
    if df.empty:
        raise ValueError("The dataset is empty. Check the CSV file content.")

    # Check label column
    if 'label' not in df.columns or df['label'].nunique() < 2:
        print("Warning: Not enough label classes for stratification. Using random split instead.")
        stratify = None
    else:
        stratify = df['label']

    # Split the data into train and validation sets
    df_train, df_val = train_test_split(df, test_size=test_size, stratify=None, random_state=random_state)

    # Save train and validation sets
    df_train.to_csv(train_csv_path, index=False)
    df_val.to_csv(val_csv_path, index=False)

    print(f"Train set saved to {train_csv_path} ({len(df_train)} samples)")
    print(f"Validation set saved to {val_csv_path} ({len(df_val)} samples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--train_csv_path", type=str, required=True, help="Path to save train CSV file.")
    parser.add_argument("--val_csv_path", type=str, required=True, help="Path to save validation CSV file.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset for validation.")
    parser.add_argument("--random_state", type=int, default=42, help="Seed for random number generator.")

    args = parser.parse_args()

    main(args.csv_path, args.train_csv_path, args.val_csv_path, args.test_size, args.random_state)
"""
import pandas as pd
from sklearn.model_selection import train_test_split

file_path = r"C:\Users\STIC-11\Desktop\sk1\train.csv"
df = pd.read_csv(file_path)
target_column = df.columns[1]
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_column])
train_csv_path = r"C:\Users\STIC-11\Desktop\class2\train.csv"
val_csv_path = r"C:\Users\STIC-11\Desktop\class2\val.csv"
train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)
print(f"✅ Data split successfully!\nTrain saved to: {train_csv_path}\nValidation saved to: {val_csv_path}")