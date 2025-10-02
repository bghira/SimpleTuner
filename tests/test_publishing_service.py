import unittest
from unittest.mock import MagicMock, patch

from simpletuner.simpletuner_sdk.server.services.publishing_service import PublishingService, PublishingServiceError


class PublishingServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.service = PublishingService()

    def test_get_license_for_model_flux(self) -> None:
        """Test license mapping for Flux models."""
        self.assertEqual(self.service.get_license_for_model("flux"), "flux-1-dev-non-commercial-license")
        self.assertEqual(self.service.get_license_for_model("FLUX"), "flux-1-dev-non-commercial-license")
        self.assertEqual(self.service.get_license_for_model("Flux-Dev"), "flux-1-dev-non-commercial-license")

    def test_get_license_for_model_sdxl(self) -> None:
        """Test license mapping for SDXL models."""
        self.assertEqual(self.service.get_license_for_model("sdxl"), "creativeml-openrail-m")
        self.assertEqual(self.service.get_license_for_model("SDXL"), "creativeml-openrail-m")
        self.assertEqual(self.service.get_license_for_model("sd-xl"), "creativeml-openrail-m")

    def test_get_license_for_model_sd1x(self) -> None:
        """Test license mapping for SD1.x models."""
        self.assertEqual(self.service.get_license_for_model("sd1x"), "creativeml-openrail-m")
        self.assertEqual(self.service.get_license_for_model("sd1.5"), "creativeml-openrail-m")
        self.assertEqual(self.service.get_license_for_model("sd-1-5"), "creativeml-openrail-m")

    def test_get_license_for_model_sd2x(self) -> None:
        """Test license mapping for SD2.x models."""
        self.assertEqual(self.service.get_license_for_model("sd2x"), "creativeml-openrail-m")
        self.assertEqual(self.service.get_license_for_model("sd2.1"), "creativeml-openrail-m")
        self.assertEqual(self.service.get_license_for_model("sd-2-1"), "creativeml-openrail-m")

    def test_get_license_for_model_sd3(self) -> None:
        """Test license mapping for SD3 models."""
        self.assertEqual(self.service.get_license_for_model("sd3"), "stabilityai-ai-community")
        self.assertEqual(self.service.get_license_for_model("SD3"), "stabilityai-ai-community")
        self.assertEqual(self.service.get_license_for_model("sd-3"), "stabilityai-ai-community")

    def test_get_license_for_model_default(self) -> None:
        """Test default license for unknown models."""
        self.assertEqual(self.service.get_license_for_model("unknown"), "apache-2.0")
        self.assertEqual(self.service.get_license_for_model("custom-model"), "apache-2.0")
        self.assertEqual(self.service.get_license_for_model(""), "apache-2.0")
        self.assertEqual(self.service.get_license_for_model(None), "apache-2.0")

    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfFolder.get_token")
    @patch("pathlib.Path.exists")
    def test_validate_token_no_token_found(self, mock_exists, mock_get_token) -> None:
        """Test token validation when no token is found."""
        mock_get_token.return_value = None
        mock_exists.return_value = False

        result = self.service.validate_token()
        self.assertFalse(result["valid"])
        self.assertIn("No HuggingFace token found", result["message"])

    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfFolder.get_token")
    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfApi")
    def test_validate_token_valid(self, mock_api_class, mock_get_token) -> None:
        """Test token validation with a valid token."""
        mock_get_token.return_value = "valid-token"
        mock_api_instance = MagicMock()
        mock_api_instance.whoami.return_value = {"name": "testuser"}
        mock_api_class.return_value = mock_api_instance
        self.service._api = mock_api_instance

        result = self.service.validate_token()
        self.assertTrue(result["valid"])
        self.assertEqual(result["username"], "testuser")

    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfFolder.get_token")
    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfApi")
    def test_check_repository_exists(self, mock_api_class, mock_get_token) -> None:
        """Test checking a repository that exists."""
        mock_get_token.return_value = "valid-token"
        mock_api_instance = MagicMock()
        mock_repo_info = MagicMock()
        mock_repo_info.id = "user/test-repo"
        mock_repo_info.private = False
        mock_api_instance.repo_info.return_value = mock_repo_info
        mock_api_class.return_value = mock_api_instance
        self.service._api = mock_api_instance

        result = self.service.check_repository("user/test-repo")
        self.assertTrue(result["exists"])
        self.assertFalse(result["available"])

    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfFolder.get_token")
    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfApi")
    def test_check_repository_available(self, mock_api_class, mock_get_token) -> None:
        """Test checking a repository that doesn't exist (available)."""
        mock_get_token.return_value = "valid-token"
        mock_api_instance = MagicMock()
        mock_api_instance.repo_info.side_effect = Exception("Not found")
        mock_api_class.return_value = mock_api_instance
        self.service._api = mock_api_instance

        result = self.service.check_repository("user/new-repo")
        self.assertFalse(result["exists"])
        self.assertTrue(result["available"])

    def test_check_repository_invalid_format(self) -> None:
        """Test checking a repository with invalid format."""
        with self.assertRaises(PublishingServiceError) as ctx:
            self.service.check_repository("invalid-repo-id")
        self.assertEqual(ctx.exception.status_code, 400)

    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfFolder.get_token")
    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfApi")
    def test_get_user_organizations(self, mock_api_class, mock_get_token) -> None:
        """Test getting user organizations."""
        mock_get_token.return_value = "valid-token"
        mock_api_instance = MagicMock()
        mock_api_instance.whoami.return_value = {
            "name": "testuser",
            "orgs": [{"name": "org1"}, {"name": "org2"}],
        }
        mock_api_class.return_value = mock_api_instance
        self.service._api = mock_api_instance

        result = self.service.get_user_organizations()
        self.assertEqual(result["username"], "testuser")
        self.assertEqual(result["organizations"], ["org1", "org2"])
        self.assertEqual(result["namespaces"], ["testuser", "org1", "org2"])

    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfFolder.get_token")
    @patch("pathlib.Path.exists")
    def test_get_organizations_no_token(self, mock_exists, mock_get_token) -> None:
        """Test getting organizations when no token exists."""
        mock_get_token.return_value = None
        mock_exists.return_value = False

        with self.assertRaises(PublishingServiceError) as ctx:
            self.service.get_user_organizations()

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("No HuggingFace token found", ctx.exception.message)

    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfFolder.get_token")
    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfApi")
    def test_get_organizations_no_orgs(self, mock_api_class, mock_get_token) -> None:
        """Test getting organizations when user has no organizations."""
        mock_get_token.return_value = "valid-token"
        mock_api_instance = MagicMock()
        mock_api_instance.whoami.return_value = {"name": "solouser", "orgs": []}
        mock_api_class.return_value = mock_api_instance
        self.service._api = mock_api_instance

        result = self.service.get_user_organizations()
        self.assertEqual(result["username"], "solouser")
        self.assertEqual(result["organizations"], [])
        self.assertEqual(result["namespaces"], ["solouser"])

    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfFolder.get_token")
    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfApi")
    def test_get_organizations_missing_orgs_key(self, mock_api_class, mock_get_token) -> None:
        """Test getting organizations when orgs key is missing from API response."""
        mock_get_token.return_value = "valid-token"
        mock_api_instance = MagicMock()
        mock_api_instance.whoami.return_value = {"name": "testuser"}
        mock_api_class.return_value = mock_api_instance
        self.service._api = mock_api_instance

        result = self.service.get_user_organizations()
        self.assertEqual(result["username"], "testuser")
        self.assertEqual(result["organizations"], [])
        self.assertEqual(result["namespaces"], ["testuser"])

    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfFolder.get_token")
    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfApi")
    def test_validate_token_invalid(self, mock_api_class, mock_get_token) -> None:
        """Test token validation with an invalid token."""
        mock_get_token.return_value = "invalid-token"
        mock_api_instance = MagicMock()
        mock_api_instance.whoami.side_effect = Exception("Invalid token")
        mock_api_class.return_value = mock_api_instance
        self.service._api = mock_api_instance

        result = self.service.validate_token()
        self.assertFalse(result["valid"])
        self.assertIn("Token is invalid or expired", result["message"])

    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfFolder.get_token")
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="hf_file_token\n")
    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfApi")
    def test_validate_token_from_file(self, mock_api_class, mock_open, mock_exists, mock_get_token) -> None:
        """Test token validation reading from file when HfFolder returns None."""
        mock_get_token.return_value = None
        mock_exists.return_value = True

        mock_api_instance = MagicMock()
        mock_api_instance.whoami.return_value = {"name": "fileuser"}
        mock_api_class.return_value = mock_api_instance
        self.service._api = mock_api_instance

        result = self.service.validate_token()
        self.assertTrue(result["valid"])
        self.assertEqual(result["username"], "fileuser")
        mock_api_instance.whoami.assert_called_once_with(token="hf_file_token")

    def test_check_repository_empty_string(self) -> None:
        """Test checking repository with empty string."""
        with self.assertRaises(PublishingServiceError) as ctx:
            self.service.check_repository("")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Invalid repository ID", ctx.exception.message)

    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfFolder.get_token")
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="hf_file_token")
    @patch("simpletuner.simpletuner_sdk.server.services.publishing_service.HfApi")
    def test_check_repository_token_from_file(self, mock_api_class, mock_open, mock_exists, mock_get_token) -> None:
        """Test that check_repository reads token from file if HfFolder returns None."""
        mock_get_token.return_value = None
        mock_exists.return_value = True

        mock_api_instance = MagicMock()
        mock_repo_info = MagicMock()
        mock_repo_info.id = "testuser/repo"
        mock_repo_info.private = True
        mock_api_instance.repo_info.return_value = mock_repo_info
        mock_api_class.return_value = mock_api_instance
        self.service._api = mock_api_instance

        result = self.service.check_repository("testuser/repo")
        self.assertTrue(result["exists"])
        self.assertFalse(result["available"])
        self.assertTrue(result["private"])

    def test_api_property_lazy_initialization(self) -> None:
        """Test that API is lazily initialized."""
        service = PublishingService()
        self.assertIsNone(service._api)

        # Access api property should initialize it
        api = service.api
        self.assertIsNotNone(service._api)
        self.assertIs(api, service._api)

        # Subsequent access should return same instance
        api2 = service.api
        self.assertIs(api, api2)

    def test_license_mapping_case_insensitive(self) -> None:
        """Test that license mapping is case-insensitive."""
        test_cases = [
            ("FLUX", "flux-1-dev-non-commercial-license"),
            ("Flux", "flux-1-dev-non-commercial-license"),
            ("FLuX", "flux-1-dev-non-commercial-license"),
            ("SDXL", "creativeml-openrail-m"),
            ("SdXl", "creativeml-openrail-m"),
            ("SD3", "stabilityai-ai-community"),
            ("Sd3", "stabilityai-ai-community"),
        ]

        for model_family, expected_license in test_cases:
            self.assertEqual(
                self.service.get_license_for_model(model_family),
                expected_license,
                f"Failed for model_family: {model_family}",
            )


if __name__ == "__main__":
    unittest.main()
