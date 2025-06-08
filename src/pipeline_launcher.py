import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.session import Session
from sagemaker.experiments.experiment import Experiment
from sagemaker.experiments.run import Run
from sagemaker import ModelPackage
import boto3
import time
import os
import secrets

session = sagemaker.Session()
role = os.environ.get('SAGEMAKER_ROLE')
bucket = session.default_bucket()

data_location = session.upload_data(
    "data/train_clientes_sample.csv",
    bucket=bucket,
    key_prefix="data"
)

experiment_name = f"recomendador-experimento2-{secrets.token_hex(4)}"
timestamp = str(int(time.time()))

experiment = Experiment.create(
    experiment_name=experiment_name,
    description="Experimento de recomendador"
)

with Run(experiment_name=experiment_name,
         run_name=f"run-{timestamp}",
         sagemaker_session=session) as run:

  estimator = SKLearn(
      entry_point="src/train_script.py",
      role=role,
      instance_type="ml.m5.large",
      framework_version="0.23-1",
      py_version="py3",
      sagemaker_session=session,
      base_job_name="recomendador-train",
      hyperparameters={
          'n-estimators': 100,
          'max-depth': 10,
          'random-state': 42
      }
  )

  # Entrenamiento del modelo
  estimator.fit({"data": data_location})

  # Registrar model en Model Registry
  model_package = estimator.register(
      content_types=["text/csv"],
      response_types=["text/csv"],
      inference_instances=["ml.m5.large"],
      transform_instances=["ml.m5.large"],
      approval_status="Approved"
  )

  print("✅ Modelo registrado:", model_package.model_package_arn)

  # Deployar modelo en sagemaker
  model = ModelPackage(
      role=role,
      model_package_arn=model_package.model_package_arn,
      sagemaker_session=session
  )

  predictor = model.deploy(
      initial_instance_count=1,
      instance_type="ml.m5.large"
  )

  print("✅ Deployment completed!")
