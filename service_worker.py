from fastapi import FastAPI

# from simpletuner_sdk import parse_api_args
from simpletuner_sdk.configuration import Configuration
from simpletuner_sdk.training_host import TrainingHost
import logging

logger = logging.getLogger("SimpleTunerAPI")

config_controller = Configuration()
training_host = TrainingHost()

app = FastAPI()

#####################################################
#   configuration controller for argument handling  #
#####################################################
app.include_router(config_controller.router)

#####################################################
#   traininghost controller for training job mgmt   #
#####################################################
app.include_router(training_host.router)
