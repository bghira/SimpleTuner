import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from simpletuner.helpers.configuration.registry import ConfigRegistry, ConfigRule, ConfigValidator, RuleType
from simpletuner.helpers.models.all import model_families


@dataclass
class ConfigDocumentationGenerator:
    """Generates documentation from configuration registry."""

    def generate_all_docs(self) -> Dict[str, Any]:
        """Generate documentation for all registered configurations."""
        docs = {
            "models": {},
            "dataloader": self._generate_category_docs("dataloader"),
            "base": self._generate_category_docs("base"),
            "all_fields": self._collect_all_fields(),
        }

        # Generate docs for each model family
        for model_family in ConfigRegistry.get_all_categories():
            if model_family not in ["dataloader", "base"]:
                docs["models"][model_family] = self._generate_category_docs(model_family)

        return docs

    def _generate_category_docs(self, category: str) -> Dict[str, Any]:
        """Generate documentation for a specific category."""
        rules = ConfigRegistry.get_rules(category)
        validators = ConfigRegistry.get_validators(category)
        category_doc = ConfigRegistry.get_documentation(category)

        # Group rules by type
        grouped_rules = self._group_rules_by_type(rules)

        return {
            "description": category_doc,
            "defaults": grouped_rules.get(RuleType.DEFAULT, []),
            "required": grouped_rules.get(RuleType.REQUIRED, []),
            "constraints": grouped_rules.get(RuleType.MIN, []) + grouped_rules.get(RuleType.MAX, []),
            "choices": grouped_rules.get(RuleType.CHOICES, []),
            "overrides": grouped_rules.get(RuleType.OVERRIDE, []),
            "combinations": grouped_rules.get(RuleType.COMBINATION, []),
            "incompatible": grouped_rules.get(RuleType.INCOMPATIBLE, []),
            "custom_validators": [{"doc": v.doc, "examples": v.examples} for v in validators],
            "examples": self._generate_examples(rules),
            "common_errors": self._generate_common_errors(rules),
        }

    def _group_rules_by_type(self, rules: List[ConfigRule]) -> Dict[RuleType, List[ConfigRule]]:
        """Group rules by their type."""
        grouped = {}
        for rule in rules:
            if rule.rule_type not in grouped:
                grouped[rule.rule_type] = []
            grouped[rule.rule_type].append(rule)
        return grouped

    def _collect_all_fields(self) -> Dict[str, List[str]]:
        """Collect all unique field names across all categories."""
        all_fields = {}
        for category in ConfigRegistry.get_all_categories():
            rules = ConfigRegistry.get_rules(category)
            for rule in rules:
                if rule.field_name not in all_fields:
                    all_fields[rule.field_name] = []
                if category not in all_fields[rule.field_name]:
                    all_fields[rule.field_name].append(category)
        return all_fields

    def _generate_examples(self, rules: List[ConfigRule]) -> List[Dict[str, str]]:
        """Generate example configurations from rules."""
        examples = []
        example_config = {}

        # Build example config from rules
        for rule in rules:
            if rule.example:
                # Parse example if it's in key: value format
                if ":" in rule.example:
                    key, value = rule.example.split(":", 1)
                    example_config[key.strip()] = value.strip()

        if example_config:
            examples.append({"title": "Example Configuration", "config": example_config})

        # Add examples from validators
        for rule in rules:
            if rule.example and "\n" in rule.example:
                examples.append({"title": f"Example for {rule.field_name}", "config": rule.example})

        return examples

    def _generate_common_errors(self, rules: List[ConfigRule]) -> List[Dict[str, str]]:
        """Generate common error scenarios from rules."""
        errors = []

        for rule in rules:
            if rule.rule_type == RuleType.REQUIRED:
                errors.append(
                    {
                        "error": f"Missing required field: {rule.field_name}",
                        "solution": rule.suggestion or f"Add {rule.field_name} to your configuration",
                        "example": rule.example,
                    }
                )
            elif rule.rule_type == RuleType.CHOICES:
                errors.append(
                    {
                        "error": f"Invalid value for {rule.field_name}",
                        "solution": f"Use one of: {', '.join(map(str, rule.value))}",
                        "example": rule.example,
                    }
                )

        return errors

    def to_markdown(self) -> str:
        """Convert documentation to markdown format."""
        docs = self.generate_all_docs()
        md_lines = ["# SimpleTuner Configuration Reference\n"]

        # Add table of contents
        md_lines.append("## Table of Contents\n")
        md_lines.append("- [Base Configuration](#base-configuration)")
        md_lines.append("- [Dataloader Configuration](#dataloader-configuration)")
        md_lines.append("- [Model-Specific Configuration](#model-specific-configuration)")
        for model in sorted(docs["models"].keys()):
            md_lines.append(f"  - [{model.upper()}](#{model}-configuration)")
        md_lines.append("- [Field Reference](#field-reference)\n")

        # Base configuration
        if docs.get("base"):
            md_lines.extend(self._category_to_markdown("Base Configuration", docs["base"]))

        # Dataloader configuration
        if docs.get("dataloader"):
            md_lines.extend(self._category_to_markdown("Dataloader Configuration", docs["dataloader"]))

        # Model-specific configurations
        md_lines.append("## Model-Specific Configuration\n")
        for model_name, model_docs in sorted(docs["models"].items()):
            md_lines.extend(self._category_to_markdown(f"{model_name.upper()} Configuration", model_docs))

        # Field reference
        md_lines.append("## Field Reference\n")
        md_lines.append("Quick reference showing which configurations use each field:\n")
        for field, categories in sorted(docs["all_fields"].items()):
            md_lines.append(f"- **{field}**: {', '.join(categories)}")

        return "\n".join(md_lines)

    def _category_to_markdown(self, title: str, category_docs: Dict[str, Any]) -> List[str]:
        """Convert a category's documentation to markdown lines."""
        lines = [f"## {title}\n"]

        if category_docs.get("description"):
            lines.append(category_docs["description"] + "\n")

        # Required fields
        if category_docs.get("required"):
            lines.append("### Required Fields\n")
            for rule in category_docs["required"]:
                lines.append(f"#### `{rule.field_name}`")
                lines.append(f"- **Description**: {rule.message}")
                if rule.suggestion:
                    lines.append(f"- **Note**: {rule.suggestion}")
                if rule.example:
                    lines.append(f"- **Example**:")
                    lines.append(f"  ```yaml")
                    lines.append(f"  {rule.example}")
                    lines.append(f"  ```")
                lines.append("")

        # Default values
        if category_docs.get("defaults"):
            lines.append("### Default Values\n")
            lines.append("| Field | Default | Description |")
            lines.append("|-------|---------|-------------|")
            for rule in category_docs["defaults"]:
                lines.append(f"| `{rule.field_name}` | `{rule.value}` | {rule.message} |")
            lines.append("")

        # Constraints
        if category_docs.get("constraints"):
            lines.append("### Constraints\n")
            for rule in category_docs["constraints"]:
                constraint_type = "Minimum" if rule.rule_type == RuleType.MIN else "Maximum"
                lines.append(f"- **{rule.field_name}**: {constraint_type} value is `{rule.value}`")
                lines.append(f"  - {rule.message}")
            lines.append("")

        # Choices
        if category_docs.get("choices"):
            lines.append("### Valid Choices\n")
            for rule in category_docs["choices"]:
                lines.append(f"#### `{rule.field_name}`")
                lines.append(f"- **Valid values**: {', '.join(map(lambda x: f'`{x}`', rule.value))}")
                lines.append(f"- **Description**: {rule.message}")
                if rule.example:
                    lines.append(f"- **Example**: `{rule.example}`")
                lines.append("")

        # Custom validators
        if category_docs.get("custom_validators"):
            lines.append("### Additional Validation\n")
            for validator in category_docs["custom_validators"]:
                lines.append(f"- {validator['doc']}")
                if validator["examples"]:
                    lines.append("  Examples:")
                    for example in validator["examples"]:
                        lines.append(f"  - {example}")
            lines.append("")

        # Common errors
        if category_docs.get("common_errors"):
            lines.append("### Common Errors\n")
            for error in category_docs["common_errors"]:
                lines.append(f"**Error**: {error['error']}")
                lines.append(f"**Solution**: {error['solution']}")
                if error.get("example"):
                    lines.append(f"**Example**:")
                    lines.append(f"```yaml")
                    lines.append(error["example"])
                    lines.append("```")
                lines.append("")

        return lines

    def to_json(self) -> str:
        """Convert documentation to JSON format."""
        return json.dumps(self.generate_all_docs(), indent=2)

    def generate_example_config(self, model_family: str) -> Dict[str, Any]:
        """Generate a minimal valid example configuration for a model family."""
        config = {}

        # Add base required fields
        for rule in ConfigRegistry.get_rules("base"):
            if rule.rule_type == RuleType.REQUIRED:
                config[rule.field_name] = f"<REQUIRED: {rule.message}>"
            elif rule.rule_type == RuleType.DEFAULT:
                config[rule.field_name] = rule.value

        # Add model-specific fields
        for rule in ConfigRegistry.get_rules(model_family):
            if rule.rule_type == RuleType.REQUIRED:
                config[rule.field_name] = f"<REQUIRED: {rule.message}>"
            elif rule.rule_type == RuleType.DEFAULT:
                config[rule.field_name] = rule.value
            elif rule.rule_type == RuleType.OVERRIDE:
                config[rule.field_name] = rule.value

        # Add example dataloader config
        if model_family in model_families:
            config["data_backend_config"] = "config/multidatabackend.json"

        return config
