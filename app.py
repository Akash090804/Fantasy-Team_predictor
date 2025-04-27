from flask import Flask, render_template, request
import pandas as pd
import joblib

#  Initialize Flask app
app = Flask(__name__)

#  Load the models
rf_model = joblib.load('rf_model.pkl')
lgbm_model = joblib.load('lgbm_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

#  Load the player features data
import pandas as pd

# Read file as single column
player_data = pd.read_csv('ipl_fantasy_features.csv', header=None)

# Split that single string into proper columns
player_data = player_data[0].str.split(",", expand=True)

# Set first row as header
player_data.columns = player_data.iloc[0]

# Remove that first row (since it's now header)
player_data = player_data.drop(0)

# Reset index
player_data = player_data.reset_index(drop=True)

# Strip extra spaces
player_data.columns = player_data.columns.str.strip()

from team_map import team_map

#  EnsembleModel class
class EnsembleModel:
    def __init__(self, rf_model, lgbm_model, xgb_model):
        self.rf_model = rf_model
        self.lgbm_model = lgbm_model
        self.xgb_model = xgb_model

    def predict(self, X):
        pred1 = self.rf_model.predict(X)
        pred2 = self.lgbm_model.predict(X)
        pred3 = self.xgb_model.predict(X)
        final_pred = (pred1 + pred2 + pred3) / 3
        return final_pred

#  Create model object
ensemble_model = EnsembleModel(rf_model, lgbm_model, xgb_model)

#  Basic home route
@app.route('/')
def index():
    teams = list(set(team_map.values()))
    teams.sort()
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    team1 = request.form['team1']
    team2 = request.form['team2']
    
    #  Select players belonging to the two selected teams
    selected_players = [player for player, team in team_map.items() if team in [team1, team2]]
    
    #  Fetch those players' data
    selected_data = player_data[player_data['Player'].isin(selected_players)].copy()

    #   ADD TEAM info here 
    selected_data['Team'] = selected_data['Player'].map(team_map)

    #  Prepare features for model
    features_cols = ['Runs_bat', '4s_bat', '6s_bat', 'Strike Rate_x',
                     'Wickets', 'Overs', 'Maidens', 'Economy', 'Dots',
                     'Fielder_Catch', 'Fielder_Stumping', 'Fielder_Runout']
    
    X = selected_data[features_cols]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)  # convert to numbers safely
    
    #  Predict fantasy points
    preds = ensemble_model.predict(X)
    selected_data['Predicted_Points'] = preds

    #  Select top 11 players
    final_team = selected_data.sort_values(by='Predicted_Points', ascending=False).head(11)
    final_team = final_team.reset_index(drop=True)

    #  Tagging Captain, Vice Captain, Normal
    tags = ['C', 'VC'] + ['NA'] * 9
    final_team['C/VC'] = tags

    # Final Output: Player, Team, C/VC
    output = final_team[['Player', 'Team', 'C/VC']].values.tolist()

    #  Render result page
    return render_template('result.html', final_team=output)

    
if __name__ == '__main__':
    app.run(debug=True)
