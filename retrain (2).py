import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

df = pd.read_csv('dataset.csv')
y = (df['claim_status'] == 'Approved').astype(int)

encoders = {}
for col in ['document_status', 'pre_auth_status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df[['claim_amount_inr','sum_insured_inr','continuous_coverage_months','document_status','pre_auth_status']].copy()
X['claim_coverage_ratio'] = X['claim_amount_inr'] / X['sum_insured_inr'] * 100

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Small model - only 20 trees, max depth 8
model = RandomForestClassifier(
    n_estimators=20,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl','wb'))
pickle.dump(encoders, open('encoders.pkl','wb'))
pickle.dump(list(X.columns), open('columns.pkl','wb'))

meta = json.load(open('meta.json'))
meta['cat_options']['document_status'] = list(encoders['document_status'].classes_)
meta['cat_options']['pre_auth_status'] = list(encoders['pre_auth_status'].classes_)
json.dump(meta, open('meta.json','w'))

size = round(os.path.getsize('model.pkl')/1024/1024, 2)
acc = round(accuracy_score(y_test, model.predict(X_test))*100, 2)
print('Accuracy:', acc, '%')
print('Model size:', size, 'MB')
