#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from azureml.core import Workspace

subscription_id = '<your subscription id>'
resource_group = '<your resource group>'
workspace_name = '<your workspace name>'

ws = Workspace.create(name=workspace_name,
                      subscription_id=subscription_id,
                      resource_group=resource_group,
                      create_resource_group=True,
                      location='eastus')

ws.write_config()


# In[ ]:


from azureml.core import Experiment

experiment_name = 'mlflow_deploy'

experiment = Experiment(workspace=ws, name=experiment_name)


# In[ ]:


import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv")

X = data.drop('medv', axis=1)
y = data['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate predictions and calculate metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log the metrics to MLflow
mlflow.log_metric('mse', mse)
mlflow.log_metric('mae', mae)
mlflow.log_metric('r2', r2)

# Log the model to MLflow
mlflow.sklearn.log_model(model, "housing-model")


# In[ ]:


with mlflow.start_run():
    run_id = mlflow.active_run().info.run_id
    
    # Train and log the MLflow model
    run_id = mlflow.projects.run(uri='./', 
                                  experiment_name=experiment_name, 
                                  run_id=run_id,
                                  use_conda=False,
                                  entry_point='train',
                                  parameters={
                                      'alpha': 0.5
                                  })


# In[ ]:


import mlflow.azureml
from azureml.core import Model

# Register the MLflow model with Azure Machine Learning
model_uri = f"runs:/{run_id}/housing-model"
model = mlflow.register_model(model_uri=model_uri, 
                              model_name="housing-model",
                              tags={"data": "housing", "model": "linear_regression"})


# In[ ]:


import mlflow.pyfunc
import pandas as pd

def init():
    global model
    
    # Retrieve the model from the model registry
    model_path = Model.get_model_path('housing-model')
    model = mlflow.pyfunc.load_model(model_path)

def run(raw_data):
    # Convert the raw data to a pandas DataFrame
    data = pd.read_json(raw_data, orient='records')

    # Generate predictions using the model
    predictions = model.predict(data)

    # Return the predictions as a dictionary
    return predictions.tolist()


# In[ ]:


from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environment('housing-env')

# Define the Conda dependencies
cd = CondaDependencies.create(conda_packages=['scikit-learn', 'numpy', 'pandas'])
env.python.conda_dependencies = cd


# In[ ]:


from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(entry_script='score.py',
                                   environment=env)


# In[ ]:


from azureml.core.webservice import AciWebservice, Webservice

deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)


# In[ ]:


service_name = 'housing-service'

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config)

service.wait_for_deployment(show_output=True)

print(service.scoring_uri)


# In[ ]:




