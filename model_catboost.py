import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier

data = pd.read_excel('bechdel.xlsx')
data.dropna(inplace=True)

features = data.drop(['year', 'imdb','code', 'binary'],1)
labels = data['binary']

encoder = LabelEncoder()
normalizer = MinMaxScaler()

features = normalizer.fit_transform(features)
labels = encoder.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)

model = CatBoostClassifier(iterations=5000,
                           learning_rate=0.02,
                           max_depth=1)

model.fit(x_train, y_train)
pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)

print(acc)
print(classification_report(y_test, pred))

cm = confusion_matrix(y_test, pred)
print(cm)

results = cross_val_score(estimator=model, X=features, y=labels,
                          cv=10, scoring='accuracy')

mean = results.mean()
std = results.std()
print(mean)
print(std)

new_data = pd.read_csv('test.csv')
new_data.dropna(inplace=True)

new_features = new_data.drop(['year', 'imdb'],1)
new_features = normalizer.fit_transform(new_features)

pred_binary = model.predict(new_features)

new_data['mdb'] = pred_binary

new_data.to_csv('test_mdb.csv')


