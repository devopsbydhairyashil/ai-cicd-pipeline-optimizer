import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
os.makedirs('model', exist_ok=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/ci_sample.csv')
    p.add_argument('--out', default='model/model.pkl')
    args = p.parse_args()
    df = pd.read_csv(args.data)
    X = df[['duration','tests_run','failures_last_24h','changed_files']]
    y = df['failed']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train,y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test,preds))
    joblib.dump(clf, args.out)
    # Save feature importances for transparency
    with open('model/feature_importances.txt','w') as f:
        for feat,imp in zip(X.columns, clf.feature_importances_):
            f.write(f"{feat}: {imp}\n")
    print('Model saved to', args.out)

if __name__ == '__main__':
    main()
