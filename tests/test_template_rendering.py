"""Unit tests for Jinja2 template rendering using unittest."""

import unittest
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


class TemplateRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        template_dir = Path(__file__).parent.parent / "templates"
        cls.env = Environment(
            loader=FileSystemLoader([template_dir, template_dir / "partials"]),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def render_field(self, field_payload):
        template = self.env.get_template("form_field.html")
        return template.render(field=field_payload)

    def test_text_field_has_basic_attributes_and_no_x_model(self):
        rendered = self.render_field(
            {
                "id": "test_field",
                "name": "test_name",
                "type": "text",
                "label": "Test Field",
                "value": "test_value",
                "placeholder": "Enter test value",
            }
        )

        self.assertIn('id="test_field"', rendered)
        self.assertIn('name="test_name"', rendered)
        self.assertIn('value="test_value"', rendered)
        self.assertIn('placeholder="Enter test value"', rendered)
        self.assertNotIn("x-model=", rendered)
        self.assertIn("x-data=", rendered)

    def test_number_field_limits_are_rendered(self):
        rendered = self.render_field(
            {
                "id": "num_field",
                "name": "num_name",
                "type": "number",
                "label": "Number Field",
                "value": "42",
                "min": 0,
                "max": 100,
                "step": 1,
            }
        )

        self.assertIn('type="number"', rendered)
        self.assertIn('min="0"', rendered)
        self.assertIn('max="100"', rendered)
        self.assertIn('step="1"', rendered)
        self.assertNotIn("x-model=", rendered)

    def test_select_field_renders_options_without_binding(self):
        rendered = self.render_field(
            {
                "id": "select_field",
                "name": "select_name",
                "type": "select",
                "label": "Select Field",
                "value": "option2",
                "options": [
                    {"value": "option1", "label": "Option 1"},
                    {"value": "option2", "label": "Option 2"},
                    {"value": "option3", "label": "Option 3"},
                ],
            }
        )

        self.assertIn("<select", rendered)
        self.assertNotIn("x-model=", rendered)
        self.assertIn('<option value="option2" selected>', rendered)

    def test_textarea_contents_render(self):
        rendered = self.render_field(
            {
                "id": "textarea_field",
                "name": "textarea_name",
                "type": "textarea",
                "label": "Textarea Field",
                "value": "Multi\nline\ntext",
                "placeholder": "Enter text...",
            }
        )

        self.assertIn("<textarea", rendered)
        self.assertNotIn("x-model=", rendered)
        self.assertIn("Multi\nline\ntext</textarea>", rendered)

    def test_checkbox_field_checked_state(self):
        rendered = self.render_field(
            {
                "id": "checkbox_field",
                "name": "checkbox_name",
                "type": "checkbox",
                "label": "Checkbox Field",
                "value": True,
            }
        )

        self.assertIn('type="checkbox"', rendered)
        self.assertNotIn("x-model=", rendered)
        self.assertIn("checked", rendered)

    def test_unique_ids_across_multiple_fields(self):
        fields = [
            {"id": "field1", "name": "name1", "type": "text", "label": "Field 1"},
            {"id": "field2", "name": "name2", "type": "text", "label": "Field 2"},
            {"id": "field3", "name": "name3", "type": "text", "label": "Field 3"},
        ]

        rendered_fields = [self.render_field(field) for field in fields]

        for field, rendered in zip(fields, rendered_fields):
            self.assertIn(f'id="{field["id"]}"', rendered)
            self.assertIn(f'name="{field["name"]}"', rendered)

        for idx, field in enumerate(fields):
            target_id = f'id="{field["id"]}"'
            for jdx, other in enumerate(rendered_fields):
                if idx == jdx:
                    continue
                self.assertNotIn(target_id, other)

    def test_required_fields_show_indicator(self):
        rendered = self.render_field(
            {
                "id": "required_field",
                "name": "required_name",
                "type": "text",
                "label": "Required Field",
                "required": True,
            }
        )

        self.assertIn('<span class="text-danger">*</span>', rendered)
        self.assertIn("required", rendered.lower())

    def test_field_description_renders(self):
        rendered = self.render_field(
            {
                "id": "desc_field",
                "name": "desc_name",
                "type": "text",
                "label": "Field with Description",
                "description": "This is a helpful description",
            }
        )

        self.assertIn("This is a helpful description", rendered)


if __name__ == "__main__":
    unittest.main()
