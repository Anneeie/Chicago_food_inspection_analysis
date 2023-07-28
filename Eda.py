import pandas as pd
import seaborn as sns

data = pd.read_csv('Food_Inspections.csv')
print(data.isna().sum())
value_counts = data['Inspection Type'].value_counts(dropna=False)
values_to_delete = value_counts[(value_counts < 40)].index.tolist()
for value in values_to_delete:
    data.drop(data[data['Inspection Type'] == value].index, inplace=True)
data.dropna(subset=['Inspection Type'], inplace=True)
data['Results'].replace('Pass w/ Conditions', 'Pass', inplace=True)
data = data[(data['Results'] == 'Pass') | (data['Results'] == 'Fail')]
data = data.reset_index(drop=True)
data = data[(data['Risk']!='All')]
def categorize_risk(risk_level):
    if risk_level == "Risk 1 (High)":
        return 3
    elif risk_level == "Risk 2 (Medium)":
        return 2
    else:
        return 1

data['Risk'] = data['Risk'].apply(categorize_risk)
sns.countplot(data=data, x='Risk')
data['Inspection Date'] = pd.to_datetime(data['Inspection Date'])
data['Zip'].fillna(data['Zip'].mode().iloc[0], inplace=True)
data.drop(['Latitude', 'Longitude', 'Location', 'Address', 'City', 'State'], axis=1, inplace=True)
data['Facility Type'].fillna(data['Facility Type'].mode().iloc[0], inplace=True)
data.drop(['Violations'], axis=1, inplace=True)
value_counts_f = data['Facility Type'].value_counts(dropna=False)
values_to_delete_f = value_counts_f[(value_counts_f < 100)].index.tolist()
for value in values_to_delete_f:
    data.drop(data[data['Facility Type'] == value].index, inplace=True)
data.dropna(subset=['Facility Type'], inplace=True)
data['Year'] = data['Inspection Date'].dt.year
data['Month'] = data['Inspection Date'].dt.month
data['Day'] = data['Inspection Date'].dt.day
data.drop(['Inspection Date'], axis=1, inplace=True)

data.to_csv('data.csv', index=False)