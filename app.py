import streamlit as st
import pickle
import pandas as pd
import time

teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 
    'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 
    'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack',
    'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]


pipe = pickle.load(open('model.pkl', 'rb'))


st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")


st.markdown(
    """
    <style>
        /* General Styles */
        body {
            background-color: #032A33;
            font-family: 'Arial', sans-serif;
            color: #D3E4E7;
        }
        .title {
            color: #82ACAB;
            text-align: center;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 30px;
        }

        /* Container and Buttons */
        .stButton>button {
            background-color: #2A777C;
            color: #D3E4E7;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 18px;
            font-weight: bold;
            transition: background-color 0.3s ease;
            border: none;
        }
        .stButton>button:hover {
            background-color: #0B4B54;
        }

        /* Selectbox and Input Fields */
        .stSelectbox, .stNumberInput {
            background-color: #0B4B54;
            color: #D3E4E7;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px;
            border: 1px solid #82ACAB;
        }

        /* Result Box */
        .result-box {
            background-color: #2A777C;
            color: #D3E4E7;
            padding: 20px;
            border-radius: 10px;
            font-size: 18px;
            text-align: center;
            font-weight: bold;
            margin-top: 20px;
        }

        .progress-bar {
            background-color: #82ACAB;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 class='title'>IPL Win Predictor</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams), key="batting_team")

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams), key="bowling_team")

selected_city = st.selectbox('Select host city', sorted(cities), key="city")


# Inputs for target, score, overs, and wickets
target = st.number_input('Target', min_value=1, step=1, key="target")

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, step=1, key="score")

with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1, key="overs")

with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1, key="wickets")

st.markdown("---")

if st.button('Predict Probability'):
    with st.spinner('Calculating win probability...'):
        time.sleep(2)

        # Avoid division by zero error
        if overs == 0:
            crr = 0
        else:
            crr = score / overs

        runs_left = target - score
        balls_left = 120 - (overs * 6)
        remaining_wickets = 10 - wickets
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [remaining_wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.markdown(f"<div class='result-box'>🏏 {batting_team} - {round(win * 100)}% Win Probability</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'>🛡️ {bowling_team} - {round(loss * 100)}% Win Probability</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'> Current run rate : {round(crr)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'> Required run rate : {float(round(rrr))}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'> runs needed : {runs_left}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'> balls left : {int(balls_left)}</div>", unsafe_allow_html=True)



        win_percentage = int(win * 100)
        st.progress(win_percentage)