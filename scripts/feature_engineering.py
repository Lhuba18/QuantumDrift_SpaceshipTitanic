import pandas as pd
import os

def engineer_features(df):
    # Spending total
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)

    # Group ID from PassengerId
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    sizes = df['Group'].value_counts().to_dict()
    df['GroupSize'] = df['Group'].map(sizes)
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

    # Binary encode
    df['CryoSleep'] = df['CryoSleep'].astype(int)
    df['VIP']       = df['VIP'].astype(int)

    # One-hot
    cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Drop unused
    df.drop(columns=['Name','Group'], inplace=True, errors='ignore')

    return df

def main():
    train = pd.read_csv('output/cleaned_train.csv')
    test  = pd.read_csv('output/cleaned_test.csv')

    transported   = train.pop('Transported')
    train_feat    = engineer_features(train)
    test_feat     = engineer_features(test)

    train_feat['Transported'] = transported.astype(int)

    os.makedirs('output', exist_ok=True)
    train_feat.to_csv('output/engineered_train.csv', index=False)
    test_feat.to_csv('output/engineered_test.csv',  index=False)

    # plain-ASCII message
    print("[OK] Engineered data saved to output/")

if __name__ == "__main__":
    main()
