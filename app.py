from flask import Flask, render_template, jsonify, request
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
        options = list(set(dataset['team1'].dropna().tolist() + dataset['team2'].dropna().tolist()))
    elif input_type == 'umpire':
        options = list(set(dataset['umpire1'].dropna().tolist() + dataset['umpire2'].dropna().tolist()))
    elif input_type == 'match_type':
        options = dataset['match_type'].dropna().unique().tolist()
    else:
        options = []
    
    return jsonify(options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        city = request.form['city']
        venue = request.form['venue']
        neutral_venue = int(request.form['neutral_venue'])
        team1 = request.form['team1']
        team2 = request.form['team2']
        toss_winner = request.form['toss_winner']
        toss_decision = request.form['toss_decision']
        umpire1 = request.form['umpire1']
        umpire2 = request.form['umpire2']
        match_type = request.form['match_type']
        
        # Preprocess input
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
        
        # Predict using model
        prediction = model.predict(input_data)
        
        # Display result
        result = "Team 1 wins" if prediction == 1 else "Team 2 wins"
        return render_template('result.html', result=result)
    
    except Exception as e:
        return render_template('result.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
