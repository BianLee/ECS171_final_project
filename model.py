import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_and_prepare_data(file_path):

    df = pd.read_csv(file_path)
    df = df.sort_values('epoch')


    df['bull_payout'] = df['totalAmount'] / df['bullAmount']
    df['bear_payout'] = df['totalAmount'] / df['bearAmount']

    df['profitable_direction'] = np.where(df['bull_payout'] > df['bear_payout'], 'Bull', 'Bear')

    df['price_change'] = df['closePrice'] - df['lockPrice']
    df['price_change_pct'] = df['price_change'] / df['lockPrice']
    df['total_amount_change'] = df['totalAmount'].diff()
    df['bull_amount_change'] = df['bullAmount'].diff()
    df['bear_amount_change'] = df['bearAmount'].diff()

    for lag in [1, 2, 3, 4, 5]:
        df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        df[f'price_change_pct_lag_{lag}'] = df['price_change_pct'].shift(lag)
        df[f'total_amount_change_lag_{lag}'] = df['total_amount_change'].shift(lag)
        df[f'bull_amount_change_lag_{lag}'] = df['bull_amount_change'].shift(lag)
        df[f'bear_amount_change_lag_{lag}'] = df['bear_amount_change'].shift(lag)

    df = df.dropna()

    return df

def prepare_features(df):
    features = [col for col in df.columns if col.endswith('_lag_1') or col.endswith('_lag_2') or 
                col.endswith('_lag_3') or col.endswith('_lag_4') or col.endswith('_lag_5')]
    X = df[features]
    y = (df['profitable_direction'] == 'Bull').astype(int)
    return X, y

def train_and_predict(X, y, test_index):
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    X_test = X.iloc[test_index].values.reshape(1, -1)
    X_test_scaled = scaler.transform(X_test)
    
    prediction = model.predict(X_test_scaled)
    probability = model.predict_proba(X_test_scaled)[0]
    return prediction[0], probability

def main():
    file_path = 'epochs_data.csv'

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        df = load_and_prepare_data(file_path)

        X, y = prepare_features(df)

        predictions = []
        probabilities = []
        epochs = []
        start_index = max(0, len(X) - 100) 
        
        print("\nPrediction Log:")
        print("Epoch\t\tPrediction\tActual\t\tConfidence\tCorrect?")
        print("-" * 70)
        
        for i in range(start_index, len(X)):
            pred, prob = train_and_predict(X, y, i)
            predictions.append(pred)
            probabilities.append(prob)
            epochs.append(df['epoch'].iloc[i])
            
            actual = y.iloc[i]
            confidence = prob[1] if pred == 1 else prob[0]
            correct = "✓" if pred == actual else "✗"
            
            print(f"{df['epoch'].iloc[i]}\t\t{'Bull' if pred == 1 else 'Bear'}\t\t{'Bull' if actual == 1 else 'Bear'}\t\t{confidence:.4f}\t\t{correct}")


        actual = y.iloc[-len(predictions):].values
        accuracy = accuracy_score(actual, predictions)
        print(f"\nModel Accuracy: {accuracy:.2f}")
        print(classification_report(actual, predictions))


        final_model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        final_model.fit(X_scaled, y)
        joblib.dump(final_model, 'final_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        print("Final model and scaler saved.")


        last_data_point = X.iloc[-1].values.reshape(1, -1)
        last_data_point_scaled = scaler.transform(last_data_point)
        next_round_prediction = final_model.predict(last_data_point_scaled)
        next_round_probability = final_model.predict_proba(last_data_point_scaled)[0]
        confidence = next_round_probability[1] if next_round_prediction[0] == 1 else next_round_probability[0]
        print(f"\nPrediction for the next round:")
        print(f"Outcome: {'Bull' if next_round_prediction[0] == 1 else 'Bear'}")
        print(f"Confidence: {confidence:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()