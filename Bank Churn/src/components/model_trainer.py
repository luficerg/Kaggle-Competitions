from src.components.data_transformation import DataTransformation
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from ensure import ensure_annotations
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

class ModelTrainer:

    @ensure_annotations
    def __init__(self, model : str):
        self.model =  model

    @ensure_annotations
    def train(self, train: pd.DataFrame):

        trans = DataTransformation()

        train = trans.drop_duplicate(train)
        train  = trans.surname(train)

        X = train.drop('Exited', axis=1)
        y = train['Exited']

        preprocessor = trans.sklearn_pipeline(X)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the models outside the condition blocks
        best_model = XGBClassifier(**
                                    {'n_estimators': 810, 'learning_rate': 0.07921079869615913,
                                    'max_depth': 5, 'min_child_weight': 8, 
                                    'gamma': 0.27423983829634263, 'random_state': 42, 
                                    'objective': 'binary:logistic',
                                    'eval_metric': 'auc', 'n_jobs': -1})
            
            
        XGB_best = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
        
        best_model = CatBoostClassifier(**
                                        {'iterations': 830, 'learning_rate': 0.08238714339235984,
                                        'depth': 5, 'l2_leaf_reg': 0.8106903985997884, 
                                        'random_state': 42, 'verbose': 0})
            
            
        Cat_best = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('model', best_model)])
        
        # Train the final model with the best hyperparameters on the entire dataset
        best_model = LGBMClassifier(**
                                    {'n_estimators': 960, 'learning_rate': 0.031725771326186744,
                                    'max_depth': 8, 'min_child_samples': 8, 
                                    'subsample': 0.7458307885861184, 'colsample_bytree': 0.5111460378911089,
                                    'random_state': 42})
        
        LGBM_best = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
        
        if self.model == "Cat":

            Cat_best.fit(X_train, y_train)

            predictions = Cat_best.predict(X_val)

            acu = accuracy_score(y_val, predictions)
            print(acu)
                
            return acu
            

        
        elif self.model == "XGB":

            XGB_best.fit(X_train, y_train)

            predictions = XGB_best.predict(X_val)

            acu = accuracy_score(y_val, predictions)
            print(acu)
                
            return acu
        
        elif self.model == "LGBM":

            LGBM_best.fit(X_train, y_train)

            predictions = LGBM_best.predict(X_val)

            acu = accuracy_score(y_val, predictions)
            print(acu)
                
            return acu
        
        voting = VotingClassifier(estimators=[
                ('Model1', LGBM_best),
                ('Model2', XGB_best),
                ('Model3', Cat_best)
                ], voting='soft', weights = [0.5, 0.3, 0.2])
        

        voting.fit(X_train, y_train)

        predictions = voting.predict(X_val)

        acu = accuracy_score(y_val, predictions)
            
        print(acu)

        return acu

if __name__ == '__main__':
    try:
        train = pd.read_csv('artifacts\data_ingestion\\train.csv')


        model_trainer = ModelTrainer(model = "vote")
        vote = model_trainer.train(train = train)
        joblib.dump(vote, "artifacts//model_trainer//model.joblib")
    
    except Exception as e:
        raise e

