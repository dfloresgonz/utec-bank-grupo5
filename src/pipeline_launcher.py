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

metric_definitions = [{
    'Name': 'validation:accuracy',
    'Regex': 'validation:accuracy=([0-9\\.]+)'
}]

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
      metric_definitions=metric_definitions,
      base_job_name="recomendador-train",
      enable_sagemaker_metrics=True,
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
  # model = ModelPackage(
  #     role=role,
  #     model_package_arn=model_package.model_package_arn,
  #     sagemaker_session=session
  # )

  # predictor = model.deploy(
  #     initial_instance_count=1,
  #     instance_type="ml.m5.large"
  # )
  experiment_name = f"recomendador-experimento2-{secrets.token_hex(4)}"

  # Simple deployment
  try:
    # Check if endpoint exists
    session.sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    print(f"🔄 Endpoint exists, deleting and recreating: {ENDPOINT_NAME}")

    # Delete existing endpoint first
    session.sagemaker_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
    time.sleep(30)

    # Deploy new endpoint
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME
    )

  except session.sagemaker_client.exceptions.ClientError:
    # Endpoint doesn't exist, create new one
    print(f"🚀 Creating new endpoint: {ENDPOINT_NAME}")

    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME
    )

print(f"🎯 Endpoint ready: {predictor.endpoint_name}")
print(f"🧪 Experiment: {experiment_name}")
print(f"🏃 Trial: {trial_name}")

print("✅ Deployment completed!")
