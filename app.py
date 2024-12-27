import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# Add a custom title with a background color
st.markdown(
    "<h1 style='text-align: center; color: #3E8E41;'>IPL Win Predictor</h1>",
    unsafe_allow_html=True
)

# Background color and padding for the page
st.markdown("""
    <style>
        .reportview-container {
            background: #f4f4f9;
            padding: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# List of teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings', 
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load the trained model (pipeline)
pipe = pickle.load(open('model.pkl', 'rb'))

# Input fields with added styling
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams), key="batting_team")
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams), key="bowling_team")

selected_city = st.selectbox('Select host city', sorted(cities), key="city")

target = st.number_input('Target', min_value=1, max_value=500, step=1, key="target")

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0, step=1, key="score")
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1, key="overs")
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1, key="wickets")

# Apply a custom button style
button_style = """
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
            border-radius: 10px;
            padding: 10px 20px;
            width: 200px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Prediction button with styled behavior
if st.button('Predict Probability'):
    # Feature Engineering: Prepare the input data
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Create the input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr],
        'venue': [''],
        'toss_winner': [''],
        'umpire1': [''],
        'umpire2': [''],
        'toss_decision': ['']
    })

    # Make prediction using the pipeline
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Display the results with different headers for the teams
    st.markdown(f"<h2 style='text-align: center; color: #3E8E41;'>Predicted Probability</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>Batting Team ({batting_team}) - {round(win * 100)}%</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>Bowling Team ({bowling_team}) - {round(loss * 100)}%</h3>", unsafe_allow_html=True)

    # Optionally, you could also display a progress bar based on prediction
    st.progress(round(win * 100))