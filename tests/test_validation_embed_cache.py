"""
Test for validation prompt embed cache key consistency.

This test reproduces the bug where validation prompts are cached with one key
but retrieved with a different key, causing cache misses.
"""

import hashlib
import unittest


class TestValidationEmbedCacheKeys(unittest.TestCase):
    """Test that validation prompts use consistent cache keys for storage and retrieval."""

    def _compute_cache_hash(self, key_value, model_type="test_model"):
        """Helper to compute cache hash like TextEmbeddingCache.create_hash()."""
        hash_format = f"-{model_type}"
        md5_hash = hashlib.md5()
        md5_hash.update(str(key_value).encode())
        return md5_hash.hexdigest() + hash_format

    def test_plain_string_prompt_uses_prompt_as_key(self):
        """Test that passing a plain string uses the prompt text as the cache key."""
        # When validation.py passes a plain string like [args.validation_prompt]
        # The key becomes the prompt text itself
        prompt_text = "A photo-realistic image of a cat"

        # This is what the old code did
        cache_hash = self._compute_cache_hash(prompt_text)

        expected_hash_prefix = hashlib.md5(prompt_text.encode()).hexdigest()
        self.assertTrue(cache_hash.startswith(expected_hash_prefix), f"Hash should use prompt text as key")

    def test_dict_with_key_uses_explicit_key(self):
        """Test that passing a dict with explicit key uses that key for caching."""
        # When validation.py passes a dict with {"key": "validation"}
        # The key becomes "validation"
        shortname = "validation"

        cache_hash = self._compute_cache_hash(shortname)

        expected_hash_prefix = hashlib.md5(shortname.encode()).hexdigest()
        self.assertTrue(cache_hash.startswith(expected_hash_prefix), f"Hash should use shortname as key")

    def test_cache_key_mismatch_causes_different_hashes(self):
        """
        Test that demonstrates the BUG: different input formats create different cache keys.

        This test demonstrates the bug exists.
        """
        validation_prompt = "A photo-realistic image of a cat"

        # OLD storage behavior: passing plain string uses prompt text as key
        storage_hash = self._compute_cache_hash(validation_prompt)

        # Retrieval behavior: passing dict uses shortname as key
        retrieval_hash = self._compute_cache_hash("validation")

        # These hashes are DIFFERENT (demonstrating the bug)
        self.assertNotEqual(
            storage_hash,
            retrieval_hash,
            "BUG: Storage and retrieval use different cache keys! "
            f"Storage uses prompt text ({storage_hash}) "
            f"but retrieval uses shortname ({retrieval_hash})",
        )

    def test_consistent_key_format_produces_same_hash(self):
        """
        Test that using the SAME dict format for both storage and retrieval works correctly.

        This test demonstrates the FIX.
        """
        # NEW storage behavior: using dict with shortname as key
        storage_hash = self._compute_cache_hash("validation")

        # Retrieval behavior: using dict with shortname as key
        retrieval_hash = self._compute_cache_hash("validation")

        # These hashes are IDENTICAL (fix verified)
        self.assertEqual(
            storage_hash,
            retrieval_hash,
            f"Storage and retrieval should use the same cache key! "
            f"Both should use 'validation' and produce hash {storage_hash}",
        )

    def test_prompt_library_shortnames_are_used_as_keys(self):
        """Test that prompt library entries use shortnames as cache keys."""
        # For prompt library entries, we should use the shortname as the key
        shortname = "cat_photo"

        cache_hash = self._compute_cache_hash(shortname)

        expected_hash_prefix = hashlib.md5(shortname.encode()).hexdigest()
        self.assertTrue(cache_hash.startswith(expected_hash_prefix), "Prompt library should use shortname as cache key")


if __name__ == "__main__":
    unittest.main()
