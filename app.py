from flask import Flask, render_template, jsonify, request,url_for
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('C:/Users/visha/OneDrive/Desktop/NxtCatch/model.pkl')

dataset = pd.read_csv('C:/Users/visha/OneDrive/Desktop/NxtCatch/ipl-complete-dataset-20082020/matches.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_options', methods=['GET'])
def get_options():
    input_type = request.args.get('type')

    if input_type == 'city':
        options = dataset['city'].dropna().unique().tolist()
    elif input_type == 'venue':
        options = dataset['venue'].dropna().unique().tolist()
    elif input_type == 'team':
        options = dataset['team1'].dropna().unique().tolist() + dataset['team2'].dropna().unique().tolist()
        options = list(set(options))  # Remove duplicates
    elif input_type == 'umpire':
        options = dataset['umpire1'].dropna().unique().tolist() + dataset['umpire2'].dropna().unique().tolist()
        options = list(set(options))  # Remove duplicates
    elif input_type == 'match_type':
        options = dataset['match_type'].dropna().unique().tolist()
    else:
        options = []

    return jsonify(options)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    city = request.form['city']
    venue = request.form['venue']
    neutral_venue = int(request.form['neutral_venue'])  # Convert to integer
    team1 = request.form['team1']
    team2 = request.form['team2']
    toss_winner = request.form['toss_winner']
    toss_decision = request.form['toss_decision']
    umpire1 = request.form['umpire1']
    umpire2 = request.form['umpire2']
    match_type = request.form['match_type']

    # Preprocess the input data to match the format expected by the model
    # Here, create a DataFrame or array for prediction (adjust column order based on your model's expected input)
    
    input_data = pd.DataFrame({
        'city': [city],
        'venue': [venue],
        'neutral_venue': [neutral_venue],
        'team1': [team1],
        'team2': [team2],
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision],
        'umpire1': [umpire1],
        'umpire2': [umpire2],
        'match_type': [match_type]
    })

    # You need to apply the same preprocessing steps here that were done during training
    # This could include label encoding, scaling, etc.

    # Example: If you applied label encoding during training, you need to apply it here as well
    # If you used one-hot encoding or any other form of preprocessing, repeat it here
    
    # For example, if you used a LabelEncoder during training:
    # label_encoder = joblib.load('label_encoder.pkl')  # if you saved the label encoder
    # input_data['team1'] = label_encoder.transform(input_data['team1'])
    # input_data['team2'] = label_encoder.transform(input_data['team2'])
    
    # Predict the outcome using the model
    prediction = model.predict(input_data)

    # Return the prediction result
    result = "Team 1 wins" if prediction == 1 else "Team 2 wins"  # Adjust based on your model's output
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
