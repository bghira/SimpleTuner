"""Debug routes for development - only loaded when DEBUG_MODE is enabled."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse

# Use the lazy wrapper for field registry
from simpletuner.simpletuner_sdk.server.services.field_registry_wrapper import lazy_field_registry as field_registry

logger = logging.getLogger(__name__)

# Only create the router if DEBUG_MODE is enabled
if os.getenv("DEBUG_MODE", "false").lower() == "true":
    router = APIRouter(prefix="/web/debug", tags=["debug"])
    logger.warning("DEBUG ROUTES ENABLED - This should not be enabled in production!")
else:
    # Return empty router if debug mode is not enabled
    router = APIRouter()


def _convert_field_to_template_format(field: Any, config_values: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a field to the format expected by templates.
    This is a copy from web.py - should be refactored to a shared utility."""
    field_dict = {
        "name": field.name,
        "id": field.name,
        "label": field.ui_label,
        "value": config_values.get(field.name, field.default_value),
        "type": field.field_type.value.lower(),
        "placeholder": field.placeholder or "",
        "help_text": field.help_text,
    }

    # Add validation rules
    if hasattr(field, "validation_rules") and field.validation_rules:
        for rule in field.validation_rules:
            if rule.type.value == "MIN" and rule.value is not None:
                field_dict["min"] = rule.value
            elif rule.type.value == "MAX" and rule.value is not None:
                field_dict["max"] = rule.value

    # Add choices for select fields
    if hasattr(field, "choices") and field.choices:
        field_dict["choices"] = field.choices

    return field_dict


# Only add routes if DEBUG_MODE is enabled
if os.getenv("DEBUG_MODE", "false").lower() == "true":

    @router.get("/field-registry", response_class=HTMLResponse)
    async def debug_field_registry(request: Request):
        """Debug endpoint to check field registry initialization and content."""
        logger.debug("=== FIELD REGISTRY DEBUG ENDPOINT ===")

        debug_info = {
            "total_fields": len(field_registry._fields),
            "field_names": list(field_registry._fields.keys()),
            "tabs_with_field_counts": {},
            "sections_by_tab": {},
            "sample_fields": {},
            "dependencies_map": field_registry._dependencies_map,
        }

        # Count fields by tab
        for field_name, field in field_registry._fields.items():
            tab = field.tab
            if tab not in debug_info["tabs_with_field_counts"]:
                debug_info["tabs_with_field_counts"][tab] = 0
            debug_info["tabs_with_field_counts"][tab] += 1

            # Collect sections by tab
            if tab not in debug_info["sections_by_tab"]:
                debug_info["sections_by_tab"][tab] = set()
            debug_info["sections_by_tab"][tab].add(field.section)

        # Convert sets to lists for JSON serialization
        for tab in debug_info["sections_by_tab"]:
            debug_info["sections_by_tab"][tab] = sorted(list(debug_info["sections_by_tab"][tab]))

        # Get sample fields from each tab
        for tab in debug_info["tabs_with_field_counts"]:
            fields_for_tab = field_registry.get_fields_for_tab(tab)
            if fields_for_tab:
                sample_field = fields_for_tab[0]
                debug_info["sample_fields"][tab] = {
                    "name": sample_field.name,
                    "ui_label": sample_field.ui_label,
                    "field_type": sample_field.field_type.value,
                    "section": sample_field.section,
                    "default_value": sample_field.default_value,
                    "help_text": (
                        sample_field.help_text[:100] + "..." if len(sample_field.help_text) > 100 else sample_field.help_text
                    ),
                }

        # Count fields by section for the "basic" tab
        basic_fields = field_registry.get_fields_for_tab("basic")
        basic_sections = {}
        for field in basic_fields:
            section = field.section
            if section not in basic_sections:
                basic_sections[section] = []
            basic_sections[section].append(field.name)
        debug_info["basic_tab_sections"] = basic_sections

        # Generate HTML response
        html_content = f"""
        <html>
        <head><title>Field Registry Debug</title></head>
        <body>
        <h1>Field Registry Debug Information</h1>
        <h2>Summary</h2>
        <p>Total fields: {debug_info['total_fields']}</p>

        <h2>Fields by Tab</h2>
        <ul>
        {"".join([f"<li>{tab}: {count} fields</li>" for tab, count in debug_info['tabs_with_field_counts'].items()])}
        </ul>

        <h2>Sample Fields</h2>
        <pre>{json.dumps(debug_info['sample_fields'], indent=2)}</pre>

        <h2>Basic Tab Sections</h2>
        <pre>{json.dumps(debug_info['basic_tab_sections'], indent=2)}</pre>

        <h2>Dependencies Map</h2>
        <pre>{json.dumps(debug_info['dependencies_map'], indent=2)}</pre>

        <h2>Raw Debug Info</h2>
        <details>
        <summary>Click to expand</summary>
        <pre>{json.dumps(debug_info, indent=2)}</pre>
        </details>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)

    @router.get("/model-family", response_class=HTMLResponse)
    async def debug_model_family(request: Request):
        """Debug model_family field specifically."""
        try:
            from simpletuner.helpers.models.all import model_families
        except ImportError:
            model_families = {}

        try:
            # Get basic tab fields
            basic_fields = field_registry.get_fields_for_tab("basic")
            model_family_field = None

            for field in basic_fields:
                if field.name == "model_family":
                    model_family_field = field
                    break

            if not model_family_field:
                return HTMLResponse(content="<html><body><h1>model_family field not found!</h1></body></html>")

            # Debug field type
            logger.info(f"model_family field_type: {model_family_field.field_type}")
            logger.info(f"model_family field_type.value: {model_family_field.field_type.value}")
            logger.info(
                f"Comparison test: {model_family_field.field_type.value} == 'SELECT' is {model_family_field.field_type.value == 'SELECT'}"
            )
            logger.info(
                f"Comparison test: {model_family_field.field_type.value} == 'select' is {model_family_field.field_type.value == 'select'}"
            )

            # Convert to template format
            config_values = {"model_family": ""}
            converted = _convert_field_to_template_format(model_family_field, config_values)

            html = f"""
            <html>
            <body>
            <h1>Model Family Field Debug</h1>
            <h2>Field Registry Data:</h2>
            <ul>
                <li>Name: {model_family_field.name}</li>
                <li>UI Label: {model_family_field.ui_label}</li>
                <li>Field Type: {model_family_field.field_type.value}</li>
                <li>Choices: {model_family_field.choices}</li>
                <li>Choice Count: {len(model_family_field.choices) if model_family_field.choices else 0}</li>
            </ul>

            <h2>Converted Template Data:</h2>
            <pre>{json.dumps(converted, indent=2, default=str)}</pre>

            <h2>Model Families Import:</h2>
            <ul>
                <li>model_families dict: {list(model_families.keys()) if model_families else 'Empty!'}</li>
                <li>Count: {len(model_families)}</li>
            </ul>
            </body>
            </html>
            """
            return HTMLResponse(content=html)
        except Exception as e:
            logger.error(f"Error in model family debug: {e}", exc_info=True)
            return HTMLResponse(content=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>", status_code=500)

    @router.get("/num-train-epochs", response_class=HTMLResponse)
    async def debug_num_train_epochs(request: Request):
        """Debug endpoint to check num_train_epochs field configuration."""
        field = field_registry.get_field("num_train_epochs")

        if not field:
            return HTMLResponse("Field not found")

        # Get the field dict as it would be converted
        config_values = {"num_train_epochs": 1}
        field_dict = _convert_field_to_template_format(field, config_values)

        html = f"""
        <html>
        <body>
        <h1>num_train_epochs Debug</h1>
        <h2>Field Registry Data</h2>
        <pre>
Name: {field.name}
Type: {field.field_type.value}
Default: {field.default_value}
Validation Rules:
{chr(10).join(f"  - Type: {rule.type.value}, Value: {rule.value}" for rule in field.validation_rules)}
        </pre>

        <h2>Converted Field Dict</h2>
        <pre>{json.dumps(field_dict, indent=2)}</pre>

        <h2>Rendered HTML</h2>
        <input type="number"
               id="num_train_epochs"
               name="--num_train_epochs"
               value="{field_dict.get('value', '')}"
               {"min='" + str(field_dict['min']) + "'" if 'min' in field_dict else ""}
               {"max='" + str(field_dict['max']) + "'" if 'max' in field_dict else ""}>

        <h3>Test the field:</h3>
        <form>
            <input type="number"
                   id="test_epochs"
                   name="test_epochs"
                   value="1"
                   min="0"
                   max="1000">
            <p>This input has min="0" max="1000"</p>
        </form>
        </body>
        </html>
        """
        return HTMLResponse(html)

    @router.get("/select-fields", response_class=HTMLResponse)
    async def debug_select_fields(request: Request):
        """Debug endpoint to check SELECT field options."""
        try:
            # Get fields from all tabs
            all_select_fields = []
            for tab in ["basic", "model", "training", "advanced", "validation"]:
                fields = field_registry.get_fields_for_tab(tab)
                for field in fields:
                    if hasattr(field, "field_type") and field.field_type.value.upper() == "SELECT":
                        field_info = {
                            "tab": tab,
                            "name": field.name,
                            "label": field.ui_label,
                            "choices": field.choices,
                            "choice_count": len(field.choices) if field.choices else 0,
                        }
                        all_select_fields.append(field_info)

            html = f"""
            <html>
            <body>
            <h1>SELECT Field Debug</h1>
            <table border="1">
            <tr><th>Tab</th><th>Field Name</th><th>Label</th><th>Choice Count</th><th>First 3 Choices</th></tr>
            {"".join([f"<tr><td>{f['tab']}</td><td>{f['name']}</td><td>{f['label']}</td><td>{f['choice_count']}</td><td>{str(f['choices'][:3]) if f['choices'] else 'None'}</td></tr>" for f in all_select_fields])}
            </table>
            <h2>Sample Field Details</h2>
            <pre>{json.dumps(all_select_fields[:3], indent=2, default=str)}</pre>
            </body>
            </html>
            """
            return HTMLResponse(content=html)
        except Exception as e:
            logger.error(f"Error in select fields debug: {e}", exc_info=True)
            return HTMLResponse(content=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>", status_code=500)

    @router.get("/test-basic-fields", response_class=HTMLResponse)
    async def debug_test_basic_fields(request: Request):
        """Quick test endpoint to verify basic tab field loading."""
        logger.debug("=== TESTING BASIC FIELDS ENDPOINT ===")

        try:
            # Test the field registry directly
            logger.debug(f"field_registry object: {field_registry}")
            logger.debug(f"field_registry._fields exists: {hasattr(field_registry, '_fields')}")
            if hasattr(field_registry, "_fields"):
                logger.debug(f"field_registry._fields length: {len(field_registry._fields)}")
                logger.debug(f"field_registry._fields keys: {list(field_registry._fields.keys())[:10]}")

            basic_fields = field_registry.get_fields_for_tab("basic")
            logger.debug(f"Direct field registry call returned {len(basic_fields)} fields")

            if basic_fields:
                sample_field = basic_fields[0]
                logger.debug(f"First field: {sample_field.name} ({sample_field.field_type.value}) - {sample_field.ui_label}")

            # Test field conversion
            config_values = {}
            converted_fields = [_convert_field_to_template_format(field, config_values) for field in basic_fields]
            logger.debug(f"Converted {len(converted_fields)} fields successfully")

            # Also check all tabs
            all_tabs = ["basic", "model", "training", "advanced", "validation"]
            tab_counts = {}
            for tab in all_tabs:
                tab_fields = field_registry.get_fields_for_tab(tab)
                tab_counts[tab] = len(tab_fields)

            html = f"""
            <html>
            <body>
            <h1>Basic Fields Test</h1>
            <p>Registry object exists: {field_registry is not None}</p>
            <p>Registry has _fields: {hasattr(field_registry, '_fields')}</p>
            <p>Total fields in registry: {len(field_registry._fields) if hasattr(field_registry, '_fields') else 'N/A'}</p>
            <hr>
            <p>Basic tab returned: {len(basic_fields)} fields</p>
            <p>Converted: {len(converted_fields)} fields</p>

            <h2>Fields by Tab:</h2>
            <ul>
            {"".join([f"<li>{tab}: {count} fields</li>" for tab, count in tab_counts.items()])}
            </ul>

            <h2>First 5 Basic Fields:</h2>
            <ul>
            {"".join([f"<li>{field.name} ({field.field_type.value}) - {field.ui_label}</li>" for field in basic_fields[:5]])}
            </ul>

            <h2>Converted Fields Sample:</h2>
            <pre>{json.dumps(converted_fields[:3], indent=2)}</pre>
            </body>
            </html>
            """

            return HTMLResponse(html)
        except Exception as e:
            logger.error(f"Error in test basic fields endpoint: {e}", exc_info=True)
            import traceback

            return HTMLResponse(
                f"<html><body><h1>Error</h1><pre>{str(e)}\n\n{traceback.format_exc()}</pre></body></html>", status_code=500
            )
