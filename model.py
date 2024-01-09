from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

df = pd.read_csv("templates/Salary_Data.csv")
df.dropna(inplace=True)
df.drop_duplicates()
df['Gender'].unique()
gender_label = LabelEncoder()
df['Gender']=gender_label.fit_transform(df['Gender'])
df.head()
edu_label_encoder = LabelEncoder()
df['Education Level'] = edu_label_encoder.fit_transform(df['Education Level'])
job_title_encoder = LabelEncoder()
df['Job Title']=job_title_encoder.fit_transform(df['Job Title'])
Y = df['Salary']
X = df.drop(['Salary'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
model=RandomForestRegressor(max_depth=9, random_state=0)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)