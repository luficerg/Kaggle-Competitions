from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation 
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
import pandas as pd
import joblib


STAGE_NAME = "Data Validation stage"
try:
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<") 

   data = pd.read_csv('artifacts\data_ingestion\\train.csv')
   data_validation = DataValidation(data, 'schema.yaml')
   data_validation.run_validation()

   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        print(e)
        raise e



STAGE_NAME = "Data Transformation stage"
try:
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<") 

   trans = DataTransformation()
   dataframe = trans.drop_duplicate(data)
   dataframe = trans.surname(dataframe)
   X_train = trans.sklearn_pipeline(dataframe)

   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        print(e)
        raise e




STAGE_NAME = "Model Trainer stage"
try:
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<") 

   model_trainer = ModelTrainer(model = "vote")
   vote = model_trainer.train(train = data)
   joblib.dump(vote, "artifacts//model_trainer//model.joblib")

   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        print(e)
        raise e



STAGE_NAME = "Model evaluation stage"
try:
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelEvaluation()
   data_ingestion.main()
   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        print(e)
        raise e






