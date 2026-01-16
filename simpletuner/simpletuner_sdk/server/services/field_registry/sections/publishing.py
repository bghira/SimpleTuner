import logging
from typing import TYPE_CHECKING

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel

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
            documentation="OPTIONS.md#--push_to_hub",
        )
    )

    registry._add_field(
        ConfigField(
            name="publishing_config",
            arg_name="--publishing_config",
            ui_label="Additional Publishing Targets",
            field_type=FieldType.TEXTAREA,
            tab="publishing",
            section="publishing_controls",
            default_value=None,
            allow_empty=True,
            placeholder='[{"provider":"s3","bucket":"my-bucket"}]',
            help_text="Optional JSON list/dict or path to a JSON file describing extra publishing providers (S3-compatible, Backblaze B2, Azure Blob, Dropbox).",
            tooltip="Accepts inline JSON, file paths, or dict-like values from the CLI. Leave blank to disable non-Hugging Face publishing.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
            documentation="OPTIONS.md#--publishing_config",
        )
    )

    registry._add_field(
        ConfigField(
            name="post_checkpoint_script",
            arg_name="--post_checkpoint_script",
            ui_label="Post-checkpoint Script",
            field_type=FieldType.TEXT,
            tab="publishing",
            section="publishing_controls",
            default_value=None,
            placeholder="/path/to/hook.sh {local_checkpoint_path}",
            help_text="Executable to run right after each checkpoint is written to disk.",
            tooltip="Supports the same placeholders as validation hooks, such as {local_checkpoint_path}, {global_step}, {tracker_run_name}, {tracker_project_name}, {model_family}, {huggingface_path}. Runs asynchronously on the main process.",
            importance=ImportanceLevel.ADVANCED,
            order=5,
            documentation="OPTIONS.md#--post_checkpoint_script",
        )
    )

    registry._add_field(
        ConfigField(
            name="post_upload_script",
            arg_name="--post_upload_script",
            ui_label="Post-upload Script",
            field_type=FieldType.TEXT,
            tab="publishing",
            section="publishing_controls",
            default_value=None,
            placeholder="/path/to/hook.sh {remote_checkpoint_path}",
            help_text="Optional executable to run after each publishing provider and Hugging Face Hub upload finishes.",
            tooltip="Supports placeholders like {remote_checkpoint_path}, {local_checkpoint_path}, {global_step}, {tracker_run_name}, {tracker_project_name}, {model_family}, {huggingface_path}. Runs asynchronously.",
            importance=ImportanceLevel.ADVANCED,
            order=6,
            documentation="OPTIONS.md#--post_upload_script",
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
            name="push_to_hub_background",
            arg_name="--push_to_hub_background",
            ui_label="Upload In Background",
            field_type=FieldType.CHECKBOX,
            tab="publishing",
            section="publishing_controls",
            default_value=False,
            dependencies=[FieldDependency(field="push_to_hub", operator="equals", value=True, action="show")],
            help_text="Send Hub uploads to a background worker so the training loop is not blocked.",
            tooltip="When enabled, checkpoint and final uploads run in a separate thread while training continues.",
            importance=ImportanceLevel.ADVANCED,
            order=3,
            documentation="OPTIONS.md#--push_to_hub_background",
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
            help_text="The name your model will have on Hugging Face Hub (e.g., 'username/my-lora'). Can be set later when uploading.",
            tooltip="If left blank, SimpleTuner derives a name from the project settings when pushing to Hub.",
            importance=ImportanceLevel.IMPORTANT,
            order=1,
            documentation="OPTIONS.md#--hub_model_id",
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

    registry._add_field(
        ConfigField(
            name="modelspec_comment",
            arg_name="--modelspec_comment",
            ui_label="Modelspec Comment",
            field_type=FieldType.TEXTAREA,
            tab="publishing",
            section="model_card",
            default_value=None,
            allow_empty=True,
            placeholder="Optional comment embedded in safetensors metadata",
            help_text="Text embedded in safetensors file metadata, visible in external model viewers. Accepts string, array of strings (joined by newlines), or {env:VAR_NAME} placeholders.",
            tooltip="Use this to add notes, version info, or training context visible in tools like ComfyUI model info.",
            importance=ImportanceLevel.ADVANCED,
            order=3,
            documentation="OPTIONS.md#--modelspec_comment",
        )
    )
