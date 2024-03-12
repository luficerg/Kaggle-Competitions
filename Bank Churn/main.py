from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation 
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


STAGE_NAME = "Data Validation stage"
try:
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataValidationTrainingPipeline()
   data_ingestion.main()
   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        print(e)
        raise e



STAGE_NAME = "Data Transformation stage"
try:
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataTransformationTrainingPipeline()
   data_ingestion.main()
   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        print(e)
        raise e




STAGE_NAME = "Model Trainer stage"
try:
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelTrainerTrainingPipeline()
   data_ingestion.main()
   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        print(e)
        raise e



STAGE_NAME = "Model evaluation stage"
try:
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelEvaluationTrainingPipeline()
   data_ingestion.main()
   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        print(e)
        raise e






