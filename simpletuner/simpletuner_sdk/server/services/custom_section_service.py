"""Service for handling custom templated sections in tabs.

This service provides a clean way to declare sections that use custom templates
instead of form fields, allowing for rich, interactive UI components within
the standard tab framework.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CustomSectionService:
    """Service for managing custom templated sections."""

    def __init__(self) -> None:
        """Initialize the custom section service."""
        self._custom_sections: Dict[str, Dict[str, Any]] = {}
        self._initialize_custom_sections()

    def _initialize_custom_sections(self) -> None:
        """Initialize all custom templated sections."""
        # Publishing tab authentication section
        self.register_custom_section(
            tab="publishing",
            section_id="authentication",
            title="Authentication",
            icon="fas fa-key",
            template="partials/authentication_section.html",
            description="HuggingFace Hub authentication settings",
        )

        # Publishing tab repository configuration section
        self.register_custom_section(
            tab="publishing",
            section_id="repository",
            title="Repository",
            icon="fas fa-cog",
            template="partials/publishing_repository_section.html",
            description=None,
        )

        # Publishing tab Discord webhooks section
        self.register_custom_section(
            tab="publishing",
            section_id="discord_webhooks",
            title="Discord Webhooks",
            icon="fas fa-bell",
            template="partials/webhooks_section.html",
            description="Configure Discord and custom webhook destinations for training notifications",
        )

    def register_custom_section(
        self,
        tab: str,
        section_id: str,
        title: str,
        template: str,
        icon: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Register a custom templated section.

        Args:
            tab: The tab this section belongs to
            section_id: Unique identifier for the section
            title: Display title for the section
            template: Template file to render for this section
            icon: Optional icon class for the section
            description: Optional description for the section
            **kwargs: Additional section properties
        """
        section_config = {
            "id": section_id,
            "title": title,
            "template": template,
            "icon": icon,
            "description": description,
            **kwargs,
        }

        key = f"{tab}:{section_id}"
        self._custom_sections[key] = section_config
        logger.debug(f"Registered custom templated section: {key}")

    def get_custom_sections_for_tab(self, tab: str) -> List[Dict[str, Any]]:
        """Get all custom templated sections for a specific tab.

        Args:
            tab: The tab to get sections for

        Returns:
            List of section configurations for the tab
        """
        sections = []
        for key, section_config in self._custom_sections.items():
            if key.startswith(f"{tab}:"):
                sections.append(section_config.copy())

        logger.debug(f"Found {len(sections)} custom sections for tab '{tab}'")
        return sections

    def merge_custom_sections_with_field_sections(
        self, tab: str, field_sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge custom templated sections with field-based sections.

        Args:
            tab: The tab to merge sections for
            field_sections: List of sections that have fields

        Returns:
            Merged list of sections with proper ordering
        """
        custom_sections = self.get_custom_sections_for_tab(tab)

        # Create a map of field sections by ID for easy lookup
        field_section_map = {section["id"]: section for section in field_sections}

        # Merge sections, with custom sections coming first
        merged_sections = []

        # Add custom sections first
        for custom_section in custom_sections:
            merged_sections.append(custom_section)

        # Add field sections that aren't overridden by custom sections
        for field_section in field_sections:
            if field_section["id"] not in [cs["id"] for cs in custom_sections]:
                merged_sections.append(field_section)

        logger.debug(
            f"Merged sections for tab '{tab}': {len(merged_sections)} total "
            f"({len(custom_sections)} custom, {len(merged_sections) - len(custom_sections)} field-based)"
        )

        return merged_sections

    def has_custom_section(self, tab: str, section_id: str) -> bool:
        """Check if a custom section exists.

        Args:
            tab: The tab to check
            section_id: The section ID to check

        Returns:
            True if the custom section exists
        """
        key = f"{tab}:{section_id}"
        return key in self._custom_sections

    def get_custom_section(self, tab: str, section_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific custom section configuration.

        Args:
            tab: The tab to get the section from
            section_id: The section ID to get

        Returns:
            The section configuration or None if not found
        """
        key = f"{tab}:{section_id}"
        return self._custom_sections.get(key)

    def list_all_custom_sections(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all custom sections organized by tab.

        Returns:
            Dict mapping tab names to lists of custom sections
        """
        result = {}
        for key, section_config in self._custom_sections.items():
            tab = key.split(":")[0]
            if tab not in result:
                result[tab] = []
            result[tab].append(section_config.copy())

        return result


# Singleton instance used by routes
CUSTOM_SECTION_SERVICE = CustomSectionService()
