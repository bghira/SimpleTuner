"""
Unit tests for Jinja2 template rendering.

Tests template components in isolation to ensure they render correctly
and don't have issues like shared Alpine.js scopes or incorrect data binding.
"""

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path


@pytest.fixture
def jinja_env():
    """Create Jinja2 environment for testing templates."""
    template_dir = Path(__file__).parent.parent / "templates"
    env = Environment(
        loader=FileSystemLoader([
            template_dir,
            template_dir / "partials"
        ]),
        autoescape=select_autoescape(['html', 'xml'])
    )
    return env


class TestFormFieldTemplate:
    """Test form_field.html template rendering."""

    def test_text_field_renders_without_x_model(self, jinja_env):
        """Test that text fields don't have x-model bindings."""
        template = jinja_env.get_template("form_field.html")

        field_data = {
            "field": {
                "id": "test_field",
                "name": "test_name",
                "type": "text",
                "label": "Test Field",
                "value": "test_value",
                "placeholder": "Enter test value"
            }
        }

        rendered = template.render(**field_data)

        # Field should render
        assert 'id="test_field"' in rendered
        assert 'name="test_name"' in rendered
        assert 'value="test_value"' in rendered
        assert 'placeholder="Enter test value"' in rendered

        # Should NOT have x-model binding
        assert 'x-model=' not in rendered

        # Should have proper Alpine.js data wrapper
        assert 'x-data=' in rendered

    def test_number_field_renders_without_x_model(self, jinja_env):
        """Test that number fields don't have x-model bindings."""
        template = jinja_env.get_template("form_field.html")

        field_data = {
            "field": {
                "id": "num_field",
                "name": "num_name",
                "type": "number",
                "label": "Number Field",
                "value": "42",
                "min": 0,
                "max": 100,
                "step": 1
            }
        }

        rendered = template.render(**field_data)

        assert 'type="number"' in rendered
        assert 'min="0"' in rendered
        assert 'max="100"' in rendered
        assert 'step="1"' in rendered
        assert 'x-model=' not in rendered

    def test_select_field_renders_without_x_model(self, jinja_env):
        """Test that select fields don't have x-model bindings."""
        template = jinja_env.get_template("form_field.html")

        field_data = {
            "field": {
                "id": "select_field",
                "name": "select_name",
                "type": "select",
                "label": "Select Field",
                "value": "option2",
                "options": [
                    {"value": "option1", "label": "Option 1"},
                    {"value": "option2", "label": "Option 2"},
                    {"value": "option3", "label": "Option 3"}
                ]
            }
        }

        rendered = template.render(**field_data)

        assert '<select' in rendered
        assert 'x-model=' not in rendered
        assert '<option value="option2" selected>' in rendered

    def test_textarea_field_renders_without_x_model(self, jinja_env):
        """Test that textarea fields don't have x-model bindings."""
        template = jinja_env.get_template("form_field.html")

        field_data = {
            "field": {
                "id": "textarea_field",
                "name": "textarea_name",
                "type": "textarea",
                "label": "Textarea Field",
                "value": "Multi\nline\ntext",
                "placeholder": "Enter text..."
            }
        }

        rendered = template.render(**field_data)

        assert '<textarea' in rendered
        assert 'x-model=' not in rendered
        assert 'Multi\nline\ntext</textarea>' in rendered

    def test_checkbox_field_renders_without_x_model(self, jinja_env):
        """Test that checkbox fields don't have x-model bindings."""
        template = jinja_env.get_template("form_field.html")

        field_data = {
            "field": {
                "id": "checkbox_field",
                "name": "checkbox_name",
                "type": "checkbox",
                "label": "Checkbox Field",
                "value": True
            }
        }

        rendered = template.render(**field_data)

        assert 'type="checkbox"' in rendered
        assert 'x-model=' not in rendered
        assert 'checked' in rendered

    def test_field_ids_and_names_are_unique(self, jinja_env):
        """Test that field IDs and names are properly set."""
        template = jinja_env.get_template("form_field.html")

        fields = [
            {"id": "field1", "name": "name1", "type": "text", "label": "Field 1"},
            {"id": "field2", "name": "name2", "type": "text", "label": "Field 2"},
            {"id": "field3", "name": "name3", "type": "text", "label": "Field 3"},
        ]

        rendered_fields = []
        for field in fields:
            rendered = template.render(field=field)
            rendered_fields.append(rendered)
            assert f'id="{field["id"]}"' in rendered
            assert f'name="{field["name"]}"' in rendered

        # Ensure no ID collision
        for i, field in enumerate(fields):
            for j, other_rendered in enumerate(rendered_fields):
                if i != j:
                    assert f'id="{field["id"]}"' not in other_rendered

    def test_required_fields_have_asterisk(self, jinja_env):
        """Test that required fields show an asterisk."""
        template = jinja_env.get_template("form_field.html")

        field_data = {
            "field": {
                "id": "required_field",
                "name": "required_name",
                "type": "text",
                "label": "Required Field",
                "required": True
            }
        }

        rendered = template.render(**field_data)

        assert '<span class="text-danger">*</span>' in rendered
        assert 'required' in rendered.lower()

    def test_field_description_renders(self, jinja_env):
        """Test that field descriptions render correctly."""
        template = jinja_env.get_template("form_field.html")

        field_data = {
            "field": {
                "id": "desc_field",
                "name": "desc_name",
                "type": "text",
                "label": "Field with Description",
                "description": "This is a helpful description"
            }
        }

        rendered = template.render(**field_data)

        assert 'This is a helpful description' in rendered
        assert 'form-text' in rendered


class TestFormSectionTemplate:
    """Test form_section.html template rendering."""

    def test_section_renders_with_fields(self, jinja_env):
        """Test that form sections render with nested fields."""
        template = jinja_env.get_template("form_section.html")

        section_data = {
            "section": {
                "id": "test-section",
                "title": "Test Section",
                "description": "This is a test section",
                "icon": "fas fa-test",
                "fields": [
                    {
                        "id": "field1",
                        "name": "field1_name",
                        "type": "text",
                        "label": "Field 1"
                    },
                    {
                        "id": "field2",
                        "name": "field2_name",
                        "type": "number",
                        "label": "Field 2"
                    }
                ]
            }
        }

        rendered = template.render(**section_data)

        # Section header should render
        assert "Test Section" in rendered
        assert "This is a test section" in rendered
        assert "fas fa-test" in rendered

        # Fields should be included
        assert 'id="field1"' in rendered
        assert 'id="field2"' in rendered

    def test_section_collapsible_behavior(self, jinja_env):
        """Test collapsible section rendering."""
        template = jinja_env.get_template("form_section.html")

        section_data = {
            "section": {
                "id": "collapsible-section",
                "title": "Collapsible Section",
                "collapsible": True,
                "expanded": False
            }
        }

        rendered = template.render(**section_data)

        # Should have click handler for collapsing
        assert '@click="expanded = !expanded"' in rendered
        assert 'cursor-pointer' in rendered
        assert 'x-show="expanded"' in rendered
        assert 'expanded: false' in rendered  # Should start collapsed

    def test_section_actions_render(self, jinja_env):
        """Test that section actions render correctly."""
        template = jinja_env.get_template("form_section.html")

        section_data = {
            "section": {
                "id": "action-section",
                "title": "Section with Actions",
                "actions": [
                    {
                        "label": "Save Changes",
                        "class": "btn-primary",
                        "icon": "fas fa-save",
                        "hx_post": "/api/save",
                        "hx_include": "#form"
                    }
                ]
            }
        }

        rendered = template.render(**section_data)

        # Action button should render
        assert "Save Changes" in rendered
        assert "btn-primary" in rendered
        assert "fas fa-save" in rendered
        assert 'hx-post="/api/save"' in rendered
        assert 'hx-include="#form"' in rendered

    def test_htmx_attributes_render_correctly(self, jinja_env):
        """Test that HTMX attributes are properly rendered."""
        template = jinja_env.get_template("form_section.html")

        section_data = {
            "section": {
                "id": "htmx-section",
                "title": "HTMX Section",
                "actions": [
                    {
                        "label": "Submit",
                        "hx_post": "/api/submit",
                        "hx_target": "#result",
                        "hx_include": ".form-fields",
                        "hx_indicator": "#spinner"
                    }
                ]
            }
        }

        rendered = template.render(**section_data)

        assert 'hx-post="/api/submit"' in rendered
        assert 'hx-target="#result"' in rendered
        assert 'hx-include=".form-fields"' in rendered
        assert 'hx-indicator="#spinner"' in rendered

    def test_nested_form_sections_isolation(self, jinja_env):
        """Test that nested form sections maintain proper isolation."""
        template = jinja_env.get_template("form_section.html")

        section1 = {
            "section": {
                "id": "section1",
                "title": "Section 1",
                "expanded": True
            }
        }

        section2 = {
            "section": {
                "id": "section2",
                "title": "Section 2",
                "expanded": False
            }
        }

        rendered1 = template.render(**section1)
        rendered2 = template.render(**section2)

        # Each should have its own Alpine.js scope
        assert 'x-data=' in rendered1
        assert 'x-data=' in rendered2

        # Each should have unique IDs
        assert 'id="section1"' in rendered1
        assert 'id="section2"' in rendered2

        # Expanded states should be independent
        assert 'expanded: true' in rendered1
        assert 'expanded: false' in rendered2


class TestTemplateIntegration:
    """Test integration between templates."""

    def test_form_fields_in_section_render_correctly(self, jinja_env):
        """Test that form fields render correctly within sections."""
        # We can't easily test the actual include without a full Flask context,
        # but we can verify the template structure
        section_template = jinja_env.get_template("form_section.html")
        field_template = jinja_env.get_template("form_field.html")

        # Verify both templates exist and can be rendered
        assert section_template is not None
        assert field_template is not None

        # Test that form_section includes the partial reference
        section_source = section_template.source
        assert "partials/form_field.html" in section_source