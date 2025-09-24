"""Field metadata API routes for dynamic UI configuration."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from simpletuner.simpletuner_sdk.server.services.field_registry_wrapper import lazy_field_registry as field_registry
try:
    from simpletuner.simpletuner_sdk.server.services.field_registry import ImportanceLevel
except ImportError:
    from enum import Enum
    class ImportanceLevel(Enum):
        ESSENTIAL = "essential"
        IMPORTANT = "important"
        ADVANCED = "advanced"
        EXPERIMENTAL = "experimental"
from simpletuner.helpers.utils.checkpoint_manager import CheckpointManager
from simpletuner.simpletuner_sdk.api_state import APIState

router = APIRouter(prefix="/api/fields", tags=["fields"])


@router.get("/metadata")
async def get_all_field_metadata() -> Dict[str, Any]:
    """Get complete field metadata for all configuration fields.

    Returns:
        Complete field metadata including fields, dependencies, and tab structure.
    """
    return field_registry.export_field_metadata()


@router.get("/tabs")
async def get_tab_structure() -> Dict[str, Any]:
    """Get the structure of all tabs and sections.

    Returns:
        Tab structure with sections and field counts.
    """
    metadata = field_registry.export_field_metadata()
    return {"tabs": metadata["tabs"]}


@router.get("/tabs/{tab_name}")
async def get_tab_fields(
    tab_name: str,
    model_family: Optional[str] = Query(None, description="Model family for filtering"),
    model_type: Optional[str] = Query(None, description="Model type (full/lora) for filtering"),
    platform: Optional[str] = Query(None, description="Platform (cuda/mps) for filtering"),
    importance_level: Optional[str] = Query("important", description="Maximum importance level to show"),
    include_advanced: bool = Query(False, description="Include advanced fields")
) -> Dict[str, Any]:
    """Get fields for a specific tab with context filtering.

    Args:
        tab_name: Name of the tab to get fields for
        model_family: Optional model family to filter fields
        model_type: Optional model type to filter fields
        platform: Optional platform to filter fields
        importance_level: Maximum importance level (essential, important, advanced, experimental)
        include_advanced: Whether to include advanced fields

    Returns:
        Fields and sections for the specified tab.
    """
    # Build context for filtering
    context = {}
    if model_family:
        context["model_family"] = model_family
    if model_type:
        context["model_type"] = model_type
    if platform:
        context["platform"] = platform

    # Get fields for tab
    fields = field_registry.get_fields_for_tab(tab_name, context)

    # Filter by importance level
    if not include_advanced and importance_level:
        try:
            max_importance = ImportanceLevel(importance_level)
            importance_order = [
                ImportanceLevel.ESSENTIAL,
                ImportanceLevel.IMPORTANT,
                ImportanceLevel.ADVANCED,
                ImportanceLevel.EXPERIMENTAL
            ]
            max_index = importance_order.index(max_importance)

            fields = [
                f for f in fields
                if importance_order.index(f.importance) <= max_index
            ]
        except ValueError:
            pass  # Invalid importance level, show all

    # Get sections for this tab
    sections = field_registry.get_sections_for_tab(tab_name)

    # Group fields by section
    fields_by_section = {}
    for field in fields:
        section_key = field.section
        if section_key not in fields_by_section:
            fields_by_section[section_key] = []
        field_data = {
            "name": field.name,
            "arg_name": field.arg_name,
            "ui_label": field.ui_label,
            "field_type": field.field_type.value,
            "default_value": field.default_value,
            "choices": field.choices,
            "help_text": field.help_text,
            "tooltip": field.tooltip,
            "placeholder": field.placeholder,
            "importance": field.importance.value,
            "warning": field.warning,
            "group": field.group,
            "order": field.order,
            "subsection": field.subsection,
            "dynamic_choices": getattr(field, 'dynamic_choices', False),
            "dependencies": [
                {
                    "field": d.field,
                    "value": d.value,
                    "values": d.values,
                    "operator": d.operator
                }
                for d in field.dependencies
            ]
        }

        # If this is the resume_from_checkpoint field and we have an output_dir in context, load checkpoints
        if field.name == "resume_from_checkpoint" and context.get("output_dir"):
            try:
                checkpoint_manager = CheckpointManager(context["output_dir"])
                checkpoints = checkpoint_manager.list_checkpoints(include_metadata=False)

                # Build dynamic choices
                dynamic_choices = [
                    {"value": "", "label": "None (Start fresh)"},
                    {"value": "latest", "label": "Latest checkpoint"}
                ]

                if checkpoints:
                    for checkpoint in checkpoints:
                        dynamic_choices.append({
                            "value": checkpoint["name"],
                            "label": f"{checkpoint['name']} (Step {checkpoint['step']})"
                        })

                field_data["choices"] = dynamic_choices
            except Exception:
                # Keep default choices if checkpoint loading fails
                pass

        fields_by_section[section_key].append(field_data)

    return {
        "tab": tab_name,
        "sections": sections,
        "fields": fields_by_section,
        "field_count": len(fields),
        "context": context
    }


@router.get("/field/{field_name}")
async def get_field_metadata(
    field_name: str,
    output_dir: Optional[str] = Query(None, description="Output directory for checkpoint loading")
) -> Dict[str, Any]:
    """Get metadata for a specific field.

    Args:
        field_name: Name of the field to get metadata for
        output_dir: Optional output directory for dynamic checkpoint loading

    Returns:
        Field metadata including validation rules and dependencies.
    """
    field = field_registry.get_field(field_name)
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field '{field_name}' not found"
        )

    field_data = {
        "name": field.name,
        "arg_name": field.arg_name,
        "ui_label": field.ui_label,
        "field_type": field.field_type.value,
        "tab": field.tab,
        "section": field.section,
        "subsection": field.subsection,
        "default_value": field.default_value,
        "choices": field.choices,
        "dynamic_choices": getattr(field, 'dynamic_choices', False),
        "validation_rules": [
            {
                "type": rule.rule_type.value,
                "value": rule.value,
                "message": rule.message,
                "condition": rule.condition
            }
            for rule in field.validation_rules
        ],
        "dependencies": [
            {
                "field": d.field,
                "value": d.value,
                "values": d.values,
                "operator": d.operator
            }
            for d in field.dependencies
        ],
        "help_text": field.help_text,
        "tooltip": field.tooltip,
        "placeholder": field.placeholder,
        "importance": field.importance.value,
        "model_specific": field.model_specific,
        "platform_specific": field.platform_specific,
        "warning": field.warning,
        "group": field.group,
        "order": field.order
    }

    # Handle dynamic checkpoint loading for resume_from_checkpoint field
    if field_name == "resume_from_checkpoint" and output_dir:
        try:
            checkpoint_manager = CheckpointManager(output_dir)
            checkpoints = checkpoint_manager.list_checkpoints(include_metadata=False)

            # Build dynamic choices
            dynamic_choices = [
                {"value": "", "label": "None (Start fresh)"},
                {"value": "latest", "label": "Latest checkpoint"}
            ]

            if checkpoints:
                for checkpoint in checkpoints:
                    dynamic_choices.append({
                        "value": checkpoint["name"],
                        "label": f"{checkpoint['name']} (Step {checkpoint['step']})"
                    })

            field_data["choices"] = dynamic_choices
        except Exception:
            # Keep default choices if checkpoint loading fails
            pass

    return field_data


@router.get("/dependencies/{field_name}")
async def get_field_dependencies(field_name: str) -> Dict[str, Any]:
    """Get fields that depend on the specified field.

    Args:
        field_name: Name of the field to get dependents for

    Returns:
        List of dependent field names.
    """
    dependents = field_registry.get_dependent_fields(field_name)
    return {
        "field": field_name,
        "dependents": dependents,
        "count": len(dependents)
    }


@router.post("/validate")
async def validate_field_value(
    field_name: str,
    value: Any,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Validate a field value against its rules.

    Args:
        field_name: Name of the field to validate
        value: Value to validate
        context: Optional context for conditional validation

    Returns:
        Validation result with any error messages.
    """
    field = field_registry.get_field(field_name)
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field '{field_name}' not found"
        )

    errors = field_registry.validate_field_value(field_name, value, context)

    return {
        "field": field_name,
        "value": value,
        "valid": len(errors) == 0,
        "errors": errors
    }


@router.get("/search")
async def search_fields(
    query: str,
    tab: Optional[str] = None,
    importance: Optional[str] = None,
    include_experimental: bool = False
) -> Dict[str, Any]:
    """Search for fields by name, label, or help text.

    Args:
        query: Search query string
        tab: Optional tab to filter by
        importance: Optional importance level to filter by
        include_experimental: Whether to include experimental fields

    Returns:
        Matching fields.
    """
    query_lower = query.lower()
    all_fields = field_registry._fields.values()

    # Filter fields
    matching_fields = []
    for field in all_fields:
        # Check if query matches
        if (query_lower in field.name.lower() or
            query_lower in field.ui_label.lower() or
            query_lower in field.help_text.lower() or
            query_lower in field.tooltip.lower() or
            query_lower in field.arg_name.lower()):

            # Apply filters
            if tab and field.tab != tab:
                continue
            if importance and field.importance.value != importance:
                continue
            if not include_experimental and field.importance == ImportanceLevel.EXPERIMENTAL:
                continue

            matching_fields.append({
                "name": field.name,
                "arg_name": field.arg_name,
                "ui_label": field.ui_label,
                "tab": field.tab,
                "section": field.section,
                "importance": field.importance.value,
                "help_text": field.help_text
            })

    return {
        "query": query,
        "results": matching_fields,
        "count": len(matching_fields)
    }


@router.get("/model-specific/{model_family}")
async def get_model_specific_fields(model_family: str) -> Dict[str, Any]:
    """Get fields specific to a model family.

    Args:
        model_family: Model family to get fields for

    Returns:
        Fields specific to the model family.
    """
    model_specific_fields = []

    for field in field_registry._fields.values():
        if field.model_specific and model_family in field.model_specific:
            model_specific_fields.append({
                "name": field.name,
                "arg_name": field.arg_name,
                "ui_label": field.ui_label,
                "tab": field.tab,
                "section": field.section,
                "importance": field.importance.value,
                "help_text": field.help_text
            })

    return {
        "model_family": model_family,
        "fields": model_specific_fields,
        "count": len(model_specific_fields)
    }


@router.get("/importance-levels")
async def get_importance_levels() -> Dict[str, Any]:
    """Get available importance levels and their descriptions.

    Returns:
        Importance levels with descriptions.
    """
    return {
        "levels": [
            {
                "value": ImportanceLevel.ESSENTIAL.value,
                "label": "Essential",
                "description": "Required fields that must be configured for training to work"
            },
            {
                "value": ImportanceLevel.IMPORTANT.value,
                "label": "Important",
                "description": "Fields that significantly affect training results"
            },
            {
                "value": ImportanceLevel.ADVANCED.value,
                "label": "Advanced",
                "description": "Fine-tuning options for experienced users"
            },
            {
                "value": ImportanceLevel.EXPERIMENTAL.value,
                "label": "Experimental",
                "description": "Bleeding edge features that may be unstable"
            }
        ]
    }