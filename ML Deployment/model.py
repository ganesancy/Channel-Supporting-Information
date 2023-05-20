import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.model_selection import StratifiedKFold
# Load the csv file
df = pd.read_excel("FDCF.xlsx")

X_OS = df[['Al2O3', 'B2O3', 'BeO', 'Ga2O3', 'GeO2', 'Li2O', "K2O", "Na2O",
           "Rb2O", "Cs2O", "SrO", "H2O", "F", "OSDA1", "OSDA2", "OH",
           "Area1", "Area2", "T", "t", "rpm"]]
Y_OS= df['CD']
cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
for train_index, test_index in cv.split(X_OS, Y_OS):
    X_Train, X_Test= X_OS.iloc[train_index], X_OS.iloc[test_index]
    Y_Train, Y_Test= Y_OS[train_index], Y_OS[test_index]

GB = GradientBoostingClassifier(random_state=10, n_estimators=40, learning_rate=0.9,
                                min_samples_split=4, )

gb_model=GB.fit(X_Train, Y_Train)

pickle.dump(gb_model, open("model.pkl", "wb"))