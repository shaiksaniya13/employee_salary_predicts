
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

data = {
    'Experience': [1, 3, 5, 7, 2, 4, 6],
    'Education': ['B.Tech', 'M.Tech', 'B.Sc', 'MCA', 'B.Tech', 'M.Tech', 'MBA'],
    'Job Role': ['Developer', 'Analyst', 'HR', 'Developer', 'Tester', 'Manager', 'HR'],
    'Location': ['Hyderabad', 'Bangalore', 'Chennai', 'Mumbai', 'Hyderabad', 'Delhi', 'Bangalore'],
    'Salary': [3.6, 6.5, 4.2, 9.0, 4.0, 7.5, 5.5]
}

df = pd.DataFrame(data)

le_edu = LabelEncoder()
le_role = LabelEncoder()
le_loc = LabelEncoder()

df['Education'] = le_edu.fit_transform(df['Education'])
df['Job Role'] = le_role.fit_transform(df['Job Role'])
df['Location'] = le_loc.fit_transform(df['Location'])

X = df.drop('Salary', axis=1)
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump((model, le_edu, le_role, le_loc), f)

print("✅ Model trained and saved to model.pkl")
