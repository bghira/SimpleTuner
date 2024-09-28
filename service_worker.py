from fastapi import FastAPI

# from simpletuner_sdk import parse_api_args
from simpletuner_sdk.configuration import Configuration
from simpletuner_sdk.training_host import TrainingHost
from fastapi.staticfiles import StaticFiles
import logging, os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger("SimpleTunerAPI")

config_controller = Configuration()
training_host = TrainingHost()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

#####################################################
#   configuration controller for argument handling  #
#####################################################
app.include_router(config_controller.router)

#####################################################
#   traininghost controller for training job mgmt   #
#####################################################
app.include_router(training_host.router)

if os.path.exists("templates/ui.template"):
    from simpletuner_sdk.interface import WebInterface

    app.include_router(WebInterface().router)
