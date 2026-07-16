from simpletuner.simpletuner_sdk.server.services.field_registry.types import ConfigField, FieldType, ImportanceLevel


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="sana_complex_human_instruction",
            arg_name="--sana_complex_human_instruction",
            ui_label="Sana Complex Human Instruction",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_specific",
            model_specific=["sana"],
            default_value="\n".join(
                [
                    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
                    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
                    "- Do not add detail that is not in the original prompt! Do not mention colors if the User Prompt does not!",
                    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
                    "- If the prompt contains explicit material, do not remove this information from the output, which would damage the performance of downstream classifiers.",
                    "Here are examples of how to transform or refine prompts:",
                    "- User Prompt: A cat sleeping -> Enhanced: A cat sleeping peacefully, showcasing the joy of pet ownership. Cute floof kitty cat gatto.",
                    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene featuring a crowd of people.",
                    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
                    "User Prompt: ",
                ]
            ),
            placeholder="complex human instruction",
            help_text="Complex human instruction for Sana model training",
            tooltip="Special instruction format for Sana model training with complex human prompts.",
            importance=ImportanceLevel.ADVANCED,
            order=33,
        )
    )
