import mlflow.pyfunc
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report,confusion_matrix, accuracy_score

import os
import re
import time
import click
import pandas as pd

import numpy as np
os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
experiment_name = "xgboost_experiment"
mlflow.set_experiment(experiment_name)


class xgboost_model(mlflow.pyfunc.PythonModel):

    def __init__(self, mindf=0,maxdf=0.95):
        self.mindf = mindf
        self.maxdf = maxdf


    def fit (self,Xtrain,ytrain):
        model = Pipeline(steps=[('vect', CountVectorizer(max_df=self.maxdf, min_df=self.mindf, max_features=1200)),
                                ('tfidf', TfidfTransformer()),
                                ('classifier', GradientBoostingClassifier(learning_rate=0.05,
                                                                          n_estimators=100,
                                                                          min_samples_split=6,
                                                                          max_depth=10, ))])
        model.fit(Xtrain,ytrain)
        self.modelo = model
        return model

    def predict(self, context, model_input):
        label = self.modelo.predict(model_input)
        prob = self.modelo.predict_proba(model_input)

        result_df = pd.DataFrame ({"label": label , "prob": [i.max()for i in prob] })
        # result = [{'label' : label[i] , 'prob' : prob[i].max()} for i in range(len(label))]
        # max_prob=
        # return {"label" : label , "prob" : max_prob}
        return result_df


@click.command(help="Trains a Text Classification model on CSV input for SupplyChain.")
@click.argument("data")
def train(data):
    df = pd.read_csv(data)
    train, test = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)
    X_train = train.text
    X_test = test.text
    xgboost=xgboost_model()
    with mlflow.start_run(run_name="xgboost_experiment") as run:
        tic = time.time()
        model_path = os.path.join('models', run.info.run_id)
        xgboost.fit(X_train,train['label'])
        duration_training = time.time() - tic
        mlflow.pyfunc.save_model(path=model_path, python_model=xgboost)
        loaded_model = mlflow.pyfunc.load_pyfunc(model_path)
        tic = time.time()

        model_output = loaded_model.predict(X_test)
        # acc = accuracy_score(test['label'],[i['label'] for i in model_output])
        class_report = classification_report(test['label'], model_output['label'], output_dict=True)

        print(model_path )

        #     ocrmodel.predict()
        duration_prediction = time.time() - tic
        mlflow.log_metric("Load Model Time", duration_training)
        mlflow.log_metric("predict Time", duration_prediction)
        # confusion_matrices = confusion_matrix(test['label'], model_output['label'])
        mlflow.log_metric("accuracy_score", class_report['accuracy'])
        mlflow.log_metric('precision', class_report['weighted avg']['precision'])
        mlflow.log_metric("recall", class_report['weighted avg']['recall'])

        mlflow.log_param('input', data)
        signature = infer_signature(X_train, loaded_model.predict(X_train))
            # mlflow.pyfunc.log_model(loaded_model, "model")
        mlflow.sklearn.log_model(loaded_model, "model", signature=signature)
        mlflow.end_run()


if __name__ == '__main__':
    train()