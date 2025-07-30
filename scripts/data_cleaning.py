import pandas as pd
import os

def clean_data(df):
    # categorical
    df['HomePlanet']   = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
    df['Destination']  = df['Destination'].fillna(df['Destination'].mode()[0])

    # cast to pandas BooleanDtype, then fill
    df['CryoSleep'] = (
        df['CryoSleep']
        .astype('boolean')   # now an ExtensionArray
        .fillna(False)       # no warning
    )
    df['VIP'] = (
        df['VIP']
        .astype('boolean')
        .fillna(False)
    )

    # if you really need numpy bool (dtype 'bool') afterwards:
    df['CryoSleep'] = df['CryoSleep'].astype(bool)
    df['VIP']      = df['VIP'].astype(bool)

    # numeric
    df['Age'] = df['Age'].fillna(df['Age'].median())
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    df[spend_cols] = df[spend_cols].fillna(0)

    # split Cabin
    df[['Deck','CabinNum','Side']] = df['Cabin'].str.split('/',expand=True)
    df.drop(columns=['Cabin'],inplace=True)
    df['CabinNum'] = (
        pd.to_numeric(df['CabinNum'],errors='coerce')
          .fillna(-1)
          .astype(int)
    )
    df['Deck'] = df['Deck'].fillna('Unknown')
    df['Side'] = df['Side'].fillna('Unknown')

    return df

def main():
    train = pd.read_csv('data/train.csv')
    test  = pd.read_csv('data/test.csv')

    train_clean = clean_data(train)
    test_clean  = clean_data(test)

    os.makedirs('output',exist_ok=True)
    train_clean.to_csv('output/cleaned_train.csv', index=False)
    test_clean.to_csv('output/cleaned_test.csv',  index=False)
    print("[OK] Cleaned data saved to output/")

if __name__=="__main__":
    main()
