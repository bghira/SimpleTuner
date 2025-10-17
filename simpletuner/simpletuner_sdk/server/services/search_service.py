"""Service for fuzzy search across tabs and fields.

This service provides intelligent search capabilities with fuzzy matching,
scoring, and ranking for tabs and configuration fields.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from unicodedata import normalize

from ..services.field_registry_wrapper import lazy_field_registry
from .tab_service import TabService

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a single search result with scoring."""

    def __init__(
        self,
        item_type: str,
        name: str,
        title: str,
        score: float,
        matched_content: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.item_type = item_type  # 'tab' or 'field'
        self.name = name
        self.title = title
        self.score = score
        self.matched_content = matched_content
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.item_type,
            "name": self.name,
            "title": self.title,
            "score": self.score,
            "matched_content": self.matched_content,
            "context": self.context,
        }


class SearchService:
    """Service for fuzzy search across tabs and fields."""

    def __init__(self, tab_service: TabService):
        """Initialize search service.

        Args:
            tab_service: TabService instance for accessing tab configurations
        """
        self.tab_service = tab_service
        self.field_registry = lazy_field_registry

    def search_tabs_and_fields(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """Search across tabs and fields with fuzzy matching.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            Dictionary containing search results
        """
        if not query or len(query.strip()) < 2:
            return {"query": query, "results": {"tabs": [], "fields": []}, "total_matches": 0}

        normalized_query = self._normalize_text(query.strip())
        if not normalized_query:
            return {"query": query, "results": {"tabs": [], "fields": []}, "total_matches": 0}

        # Get all tabs and their fields
        all_tabs = self.tab_service.get_all_tabs()
        tab_results = []
        field_results = []

        for tab_info in all_tabs:
            tab_name = tab_info["name"]
            tab_title = tab_info["title"]
            tab_description = tab_info.get("description", "")

            # Search tab metadata
            tab_score = self._calculate_match_score(
                normalized_query,
                [tab_title, tab_description, tab_name],
                weights=[0.5, 0.3, 0.2],
            )

            if tab_score > 0.1:  # Minimum threshold for tab matches
                tab_result = SearchResult(
                    item_type="tab",
                    name=tab_name,
                    title=tab_title,
                    score=tab_score,
                    matched_content=self._get_matched_content(normalized_query, tab_title),
                    context={"description": tab_description, "icon": tab_info.get("icon")},
                )
                tab_results.append(tab_result)

            # Search fields within this tab
            try:
                tab_fields = self.field_registry.get_fields_for_tab(tab_name)
                for field in tab_fields:
                    field_score = self._calculate_field_match_score(normalized_query, field, tab_name)

                    if field_score > 0.15:  # Minimum threshold for field matches
                        field_result = SearchResult(
                            item_type="field",
                            name=field.name,
                            title=getattr(field, "ui_label", field.name),
                            score=field_score,
                            matched_content=self._get_field_matched_content(normalized_query, field),
                            context={
                                "tab": tab_name,
                                "tab_title": tab_title,
                                "section": getattr(field, "section", None),
                                "description": getattr(field, "help_text", ""),
                                "field_type": getattr(field, "field_type", None),
                            },
                        )
                        field_results.append(field_result)

            except Exception as e:
                logger.warning(f"Failed to search fields for tab {tab_name}: {e}")
                continue

        # Sort results by score (descending)
        tab_results.sort(key=lambda x: x.score, reverse=True)
        field_results.sort(key=lambda x: x.score, reverse=True)

        # Apply limits
        tab_results = tab_results[:limit]
        field_results = field_results[:limit]

        return {
            "query": query,
            "results": {
                "tabs": [result.to_dict() for result in tab_results],
                "fields": [result.to_dict() for result in field_results],
            },
            "total_matches": len(tab_results) + len(field_results),
        }

    def _calculate_field_match_score(self, query: str, field: Any, tab_name: str) -> float:
        """Calculate match score for a field.

        Args:
            query: Normalized search query
            field: Field object
            tab_name: Name of the containing tab

        Returns:
            Match score between 0 and 1
        """
        # Gather searchable text from field
        searchable_parts = []

        # Field label (highest weight)
        label = getattr(field, "ui_label", "")
        if label:
            searchable_parts.append((label, 0.4))

        # Field name/ID
        field_name = getattr(field, "name", "")
        if field_name:
            searchable_parts.append((field_name, 0.25))

        # Field argument name
        arg_name = getattr(field, "arg_name", "")
        if arg_name and arg_name != field_name:
            searchable_parts.append((arg_name, 0.15))

        # Help text/description
        help_text = getattr(field, "help_text", "")
        if help_text:
            searchable_parts.append((help_text, 0.15))

        # Section name
        section = getattr(field, "section", "")
        if section:
            searchable_parts.append((section, 0.05))

        if not searchable_parts:
            return 0.0

        # Calculate weighted score
        texts = [text for text, _ in searchable_parts]
        weights = [weight for _, weight in searchable_parts]

        base_score = self._calculate_match_score(query, texts, weights)

        # Boost score for exact matches in important fields
        if label and query.lower() in label.lower():
            base_score *= 1.2

        if field_name and query.lower() in field_name.lower():
            base_score *= 1.1

        # Boost score for fields in tabs that also match
        tab_config = self.tab_service.get_tab_config(tab_name)
        if tab_config:
            tab_score = self._calculate_match_score(query, [tab_config.title, tab_config.description or ""], [0.7, 0.3])
            if tab_score > 0.2:
                base_score *= 1.05

        return min(base_score, 1.0)

    def _calculate_match_score(self, query: str, texts: List[str], weights: Optional[List[float]] = None) -> float:
        """Calculate match score for query against multiple text strings.

        Args:
            query: Normalized search query
            texts: List of text strings to match against
            weights: Optional weights for each text (default: equal weights)

        Returns:
            Match score between 0 and 1
        """
        if not query or not texts:
            return 0.0

        if weights is None:
            weights = [1.0] * len(texts)

        if len(weights) != len(texts):
            weights = [1.0] * len(texts)

        total_score = 0.0
        total_weight = 0.0

        for text, weight in zip(texts, weights):
            if not text:
                continue

            # Simple case-insensitive substring match - no fuzzy search
            if query.lower() in text.lower():
                # Exact match gets higher score
                if query.lower() == text.lower():
                    score = 1.0
                else:
                    score = 0.8
            else:
                score = 0.0  # No fuzzy matching at all

            total_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return min(total_score / total_weight, 1.0)

    def _calculate_word_bonus(self, query: str, text: str) -> float:
        """Calculate bonus score for word-level matches.

        Args:
            query: Normalized search query
            text: Normalized text to match against

        Returns:
            Bonus score between 0 and 1
        """
        query_words = query.split()
        text_words = text.split()

        if not query_words:
            return 0.0

        matches = 0
        for query_word in query_words:
            for text_word in text_words:
                if query_word in text_word or text_word in query_word:
                    matches += 1
                    break
            else:
                # Check for fuzzy word match with dynamic threshold based on query length
                for text_word in text_words:
                    similarity = SequenceMatcher(None, query_word, text_word).ratio()

                    # Use higher threshold for short queries to reduce false positives
                    if len(query_word) <= 3:
                        threshold = 0.9  # 90% similarity for 3 or fewer characters
                    elif len(query_word) <= 5:
                        threshold = 0.8  # 80% similarity for 4-5 characters
                    else:
                        threshold = 0.7  # 70% similarity for longer words

                    if similarity > threshold:
                        matches += 0.3  # Reduced from 0.5 to 0.3
                        break

        return min(matches / len(query_words), 1.0)

    def _get_matched_content(self, query: str, text: str) -> str:
        """Get the best matching portion of text for highlighting.

        Args:
            query: Normalized search query
            text: Original text

        Returns:
            Portion of text that best matches the query
        """
        if not query or not text:
            return ""

        normalized_text = self._normalize_text(text)
        if query in normalized_text:
            # Return the original text portion that matches
            start_idx = normalized_text.find(query)
            if start_idx >= 0:
                # Map back to original text (approximate)
                return text[start_idx : start_idx + len(query)]

        # Return first word that contains the query
        words = text.split()
        for word in words:
            if query.lower() in word.lower():
                return word

        return text[:50] + "..." if len(text) > 50 else text

    def _get_field_matched_content(self, query: str, field: Any) -> str:
        """Get the best matching content from a field.

        Args:
            query: Normalized search query
            field: Field object

        Returns:
            Best matching content from the field
        """
        # Try label first
        label = getattr(field, "ui_label", "")
        if label and query.lower() in label.lower():
            return label

        # Try field name
        field_name = getattr(field, "name", "")
        if field_name and query.lower() in field_name.lower():
            return field_name

        # Try help text
        help_text = getattr(field, "help_text", "")
        if help_text:
            matched = self._get_matched_content(query, help_text)
            if matched and matched != "":
                return matched

        # Fallback to label
        return label or field_name or ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for search comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Normalize Unicode characters
        normalized = normalize("NFKD", text.lower())

        # Remove diacritics and non-alphanumeric characters (except spaces)
        cleaned = re.sub(r"[^\w\s]", " ", normalized)

        # Collapse multiple spaces
        collapsed = re.sub(r"\s+", " ", cleaned).strip()

        return collapsed
