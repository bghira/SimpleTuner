from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os


class WebInterface:
    def __init__(self):
        self.router = APIRouter(prefix="/web")
        self.router.add_api_route(
            "/", self.get_page, methods=["GET"], response_class=HTMLResponse
        )
        self.templates = Jinja2Templates(
            directory="templates"
        )  # Define the directory where your templates are stored
        self.template_file = "ui.template"  # This should correspond to an HTML file in the templates directory

    async def get_page(self, request: Request):
        """
        Retrieve ui.template from disk and display it to the user.
        If the template file does not exist, display a default message.
        """
        if os.path.exists(f"templates/{self.template_file}"):
            # Serve the template if it exists
            return self.templates.TemplateResponse(
                self.template_file, {"request": request}
            )
        else:
            # Default HTML if template is missing
            return HTMLResponse(
                content="""
            <!DOCTYPE html>
            <html>
            <head>
                <title>SimpleTuner</title>
            </head>
            <body>
                <h1>SimpleTuner</h1>
                <p>Welcome to SimpleTuner. This installation does not include a compatible web interface. Sorry.</p>
            </body>
            </html>
            """,
                status_code=200,
            )
