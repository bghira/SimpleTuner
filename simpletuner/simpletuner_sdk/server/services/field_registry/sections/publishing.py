import logging

from typing import TYPE_CHECKING

from ..types import (
    ConfigField,
    FieldDependency,
    FieldType,
    ImportanceLevel,
)

if TYPE_CHECKING:
    from ..registry import FieldRegistry


logger = logging.getLogger(__name__)


def register_publishing_fields(registry: "FieldRegistry") -> None:
    """Register Hugging Face publishing configuration fields."""

    logger.debug("register_publishing_fields called")

    registry._add_field(
        ConfigField(
            name="push_to_hub",
            arg_name="--push_to_hub",
            ui_label="Push to Hugging Face Hub",
            field_type=FieldType.CHECKBOX,
            tab="publishing",
            section="publishing_controls",
            default_value=False,
            help_text="Automatically upload the trained model to your Hugging Face Hub repository.",
            tooltip="Requires a Hugging Face token. The model uploads when training completes (and optionally at checkpoints).",
            importance=ImportanceLevel.IMPORTANT,
            order=1,
        )
    )

    registry._add_field(
        ConfigField(
            name="push_checkpoints_to_hub",
            arg_name="--push_checkpoints_to_hub",
            ui_label="Push Intermediate Checkpoints",
            field_type=FieldType.CHECKBOX,
            tab="publishing",
            section="publishing_controls",
            default_value=False,
            dependencies=[FieldDependency(field="push_to_hub", operator="equals", value=True, action="show")],
            help_text="Upload intermediate checkpoints to the same Hugging Face repository during training.",
            tooltip="Enable to mirror training progress on the Hub. This will increase upload time and storage usage.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
        )
    )

    registry._add_field(
        ConfigField(
            name="hub_model_id",
            arg_name="--hub_model_id",
            ui_label="Repository ID",
            field_type=FieldType.TEXT,
            tab="publishing",
            section="repository",
            default_value=None,
            placeholder="username/model-name",
            help_text=None,
            tooltip="If left blank, SimpleTuner derives a name from the project settings when pushing to Hub.",
            importance=ImportanceLevel.IMPORTANT,
            order=1,
        )
    )

    registry._add_field(
        ConfigField(
            name="model_card_private",
            arg_name="--model_card_private",
            ui_label="Private Repository",
            field_type=FieldType.CHECKBOX,
            tab="publishing",
            section="repository",
            default_value=True,
            dependencies=[FieldDependency(field="push_to_hub", operator="equals", value=True, action="show")],
            help_text="Create the Hugging Face repository as private instead of public.",
            tooltip="Private repositories require an organization or paid plan on Hugging Face to share with collaborators.",
            importance=ImportanceLevel.IMPORTANT,
            order=2,
        )
    )

    registry._add_field(
        ConfigField(
            name="model_card_safe_for_work",
            arg_name="--model_card_safe_for_work",
            ui_label="Mark As Safe For Work",
            field_type=FieldType.CHECKBOX,
            tab="publishing",
            section="model_card",
            default_value=False,
            dependencies=[FieldDependency(field="push_to_hub", operator="equals", value=True, action="show")],
            help_text="Remove the default NSFW warning from the generated model card on Hugging Face Hub.",
            tooltip="Only disable the warning if the model cannot generate NSFW content.",
            importance=ImportanceLevel.ADVANCED,
            order=1,
        )
    )

    registry._add_field(
        ConfigField(
            name="model_card_note",
            arg_name="--model_card_note",
            ui_label="Model Card Note",
            field_type=FieldType.TEXTAREA,
            tab="publishing",
            section="model_card",
            default_value=None,
            placeholder="Add a short introduction or usage guidance",
            dependencies=[FieldDependency(field="push_to_hub", operator="equals", value=True, action="show")],
            help_text="Optional note that appears at the top of the generated model card.",
            tooltip="Use this to provide additional context, licensing information, or usage guidance for your model.",
            importance=ImportanceLevel.IMPORTANT,
            order=2,
        )
    )
