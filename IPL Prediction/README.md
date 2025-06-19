# IPL Match Winner Prediction using Random Forest

This project uses machine learning to predict the winning team in an IPL match based on key features like team1, team2, venue, toss winner, and toss decision. The model is built using the Random Forest Classifier, which handles categorical data and multi-feature decision-making effectively.

---

##  Project Structure

- `ipl prediction.ipynb` → Jupyter Notebook containing full data analysis and ML pipeline  
- `IPL Matches 2008-2020.csv` → Dataset used for training  
- `ipl.py` → (optional) Interactive web app for real-time predictions  
- `README.md` → Project documentation

---

##  Dataset Details

The dataset includes IPL match data from **2008 to 2020**, with the following relevant columns:
- `team1`, `team2` – Competing teams
- `venue` – Match location
- `toss_winner`, `toss_decision`
- `winner` – Match result (target variable)

---

##  Notebook Workflow Highlights

### 🔹 Data Cleaning & Preprocessing
- Dropped columns like `id`, `date`, `umpire1`, `umpire2`, etc.
- Filtered out old/franchise-changed teams (e.g., Kochi Tuskers)
- Removed rows with null values
- Applied **one-hot encoding** to categorical features for ML compatibility

### 🔹 Model Building
- Used `RandomForestClassifier` from `sklearn.ensemble`
- Split data into training and testing sets
- Trained model on encoded match data
- Predicted winner using team, toss, and venue data

### 🔹 Model Performance
- Evaluated using **accuracy score**
- Achieved an accuracy of **~82%** (varies slightly depending on random state)
- Tested with sample inputs for match predictions

---

##  Optional Streamlit Web App

A `ipl.py` can be created to:
- Select input values from dropdowns (team1, team2, toss winner, toss decision, venue)
- Show instant winner prediction

To run:

```bash
streamlit run ipl.py

