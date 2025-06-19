import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the trained model and encoders (simulate here for demo)
@st.cache_data
def load_model_and_encoders():
    # Load training data to fit encoders
    df = pd.read_csv("D:\\MLProjects\\Datasets\\ipl prediction.csv")
    
    final_df = df.groupby("mid").tail(1).copy()
    team_totals = df.groupby(['mid', 'batting_team'])['total'].max().reset_index()
    match_winners = team_totals.loc[team_totals.groupby('mid')['total'].idxmax()].reset_index(drop=True)
    match_winners.rename(columns={'batting_team': 'winner'}, inplace=True)
    final_df = final_df.merge(match_winners[['mid', 'winner']], on='mid')

    features = final_df[["batting_team", "bowling_team", "venue", 
                         "total", "wickets", "overs", "runs_last_5", "wickets_last_5"]]
    target = final_df["batting_team"] == final_df["winner"]

    le_team = LabelEncoder()
    le_venue = LabelEncoder()
    features["batting_team"] = le_team.fit_transform(features["batting_team"])
    features["bowling_team"] = le_team.transform(features["bowling_team"])
    features["venue"] = le_venue.fit_transform(features["venue"])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, target)

    return model, le_team, le_venue, features

# Load model and encoders
model, le_team, le_venue, feature_data = load_model_and_encoders()

# Streamlit UI
st.title("🏏 IPL Winning Team Predictor")
st.write("Enter current match details to predict if the **batting team** will win.")

# Dropdowns
teams = list(le_team.classes_)
venues = list(le_venue.classes_)

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", [team for team in teams if team != batting_team])
venue = st.selectbox("Venue", venues)

# Match Stats
total_runs = st.number_input("Total Runs Scored", min_value=0, max_value=300, value=100)
wickets = st.slider("Wickets Lost", 0, 10, 2)
overs = st.slider("Overs Completed", 0.0, 20.0, 10.0, step=0.1)
runs_last_5 = st.slider("Runs in Last 5 Overs", 0, 100, 30)
wickets_last_5 = st.slider("Wickets Lost in Last 5 Overs", 0, 5, 1)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "batting_team": le_team.transform([batting_team])[0],
        "bowling_team": le_team.transform([bowling_team])[0],
        "venue": le_venue.transform([venue])[0],
        "total": total_runs,
        "wickets": wickets,
        "overs": overs,
        "runs_last_5": runs_last_5,
        "wickets_last_5": wickets_last_5
    }])

    prediction = model.predict(input_data)[0]
    
    if prediction:
        st.success(f"🎉 Prediction: {batting_team} is likely to **WIN** the match!")
    else:
        st.error(f"💔 Prediction: {batting_team} is likely to **LOSE** the match.")
