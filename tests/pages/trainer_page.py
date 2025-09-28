"""Page object for Trainer page."""

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from .base_page import BasePage


class TrainerPage(BasePage):
    """Page object for the trainer interface."""

    # Locators
    SAVE_CONFIG_BUTTON = (
        By.CSS_SELECTOR,
        r"""button[x-on\:click="Alpine.store('trainer').saveConfig()"]""",
    )
    START_TRAINING_BUTTON = (
        By.CSS_SELECTOR,
        r"""button[x-on\:click="Alpine.store('trainer').startTraining()"]""",
    )
    STOP_TRAINING_BUTTON = (
        By.CSS_SELECTOR,
        r"""button[x-on\:click="Alpine.store('trainer').stopTraining()"]""",
    )

    # Tab buttons
    BASIC_TAB = (By.CSS_SELECTOR, ".tab-btn[data-tab='basic']")
    MODEL_TAB = (By.CSS_SELECTOR, ".tab-btn[data-tab='model']")
    TRAINING_TAB = (By.CSS_SELECTOR, ".tab-btn[data-tab='training']")
    ADVANCED_TAB = (By.CSS_SELECTOR, ".tab-btn[data-tab='advanced']")
    DATASETS_TAB = (By.CSS_SELECTOR, ".tab-btn[data-tab='datasets']")
    ENVIRONMENTS_TAB = (By.CSS_SELECTOR, ".tab-btn[data-tab='environments']")

    # Status indicators
    TRAINING_STATUS = (By.CSS_SELECTOR, '[x-show="trainingStatus"]')
    STATUS_IDLE = (By.CSS_SELECTOR, "[x-show=\"trainingStatus == 'idle'\"]")
    STATUS_RUNNING = (By.CSS_SELECTOR, "[x-show=\"trainingStatus == 'running'\"]")
    STATUS_ERROR = (By.CSS_SELECTOR, "[x-show=\"trainingStatus == 'error'\"]")

    # Configuration validation
    CONFIG_VALID_INDICATOR = (By.CSS_SELECTOR, ".config-valid")
    CONFIG_INVALID_INDICATOR = (By.CSS_SELECTOR, ".config-invalid")

    def navigate_to_trainer(self):
        """Navigate to the trainer page."""
        self.navigate_to("/web/trainer")
        # Wait for page to load
        self.wait.until(EC.presence_of_element_located(self.BASIC_TAB))

    def save_configuration(self):
        """Click the save configuration button."""
        self.click_element(*self.SAVE_CONFIG_BUTTON)
        self.wait_for_htmx()

    def start_training(self):
        """Click the start training button."""
        self.click_element(*self.START_TRAINING_BUTTON)
        self.wait_for_htmx()

    def stop_training(self):
        """Click the stop training button."""
        self.click_element(*self.STOP_TRAINING_BUTTON)
        self.wait_for_htmx()

    def is_config_valid(self):
        """Check if configuration is valid.

        Returns:
            True if valid, False if invalid
        """
        return self.is_element_visible(*self.CONFIG_VALID_INDICATOR, timeout=2)

    def get_training_status(self):
        """Get current training status.

        Returns:
            Status string: 'idle', 'running', 'error', or None
        """
        if self.is_element_visible(*self.STATUS_IDLE, timeout=1):
            return "idle"
        elif self.is_element_visible(*self.STATUS_RUNNING, timeout=1):
            return "running"
        elif self.is_element_visible(*self.STATUS_ERROR, timeout=1):
            return "error"
        return None

    def switch_to_basic_tab(self):
        """Switch to Basic Configuration tab."""
        self.click_element(*self.BASIC_TAB)
        self.wait_for_htmx()

    def switch_to_model_tab(self):
        """Switch to Model Configuration tab."""
        self.click_element(*self.MODEL_TAB)
        self.wait_for_htmx()

    def switch_to_training_tab(self):
        """Switch to Training Parameters tab."""
        self.click_element(*self.TRAINING_TAB)
        self.wait_for_htmx()

    def switch_to_advanced_tab(self):
        """Switch to Advanced Options tab."""
        self.click_element(*self.ADVANCED_TAB)
        self.wait_for_htmx()

    def switch_to_datasets_tab(self):
        """Switch to Datasets tab."""
        self.click_element(*self.DATASETS_TAB)
        self.wait_for_htmx()

    def switch_to_environments_tab(self):
        """Switch to Environments tab."""
        self.click_element(*self.ENVIRONMENTS_TAB)
        self.wait_for_htmx()


class BasicConfigTab(BasePage):
    """Page object for Basic Configuration tab."""

    # Form fields
    CONFIGS_DIR_INPUT = (By.ID, "configs_dir")
    MODEL_NAME_INPUT = (By.ID, "model_name")
    OUTPUT_DIR_INPUT = (By.ID, "output_dir")
    BASE_MODEL_INPUT = (By.ID, "pretrained_model_name_or_path")

    # Save button (specific to Basic Config)
    SAVE_BUTTON = (
        By.CSS_SELECTOR,
        r"""#tab-basic button[x-on\:click="Alpine.store('trainer').saveConfig()"]""",
    )

    def set_configs_dir(self, path):
        """Set the configurations directory."""
        self.send_keys(*self.CONFIGS_DIR_INPUT, path)

    def set_model_name(self, name):
        """Set the model name."""
        self.send_keys(*self.MODEL_NAME_INPUT, name)

    def set_output_dir(self, path):
        """Set the output directory."""
        self.send_keys(*self.OUTPUT_DIR_INPUT, path)

    def set_base_model(self, model):
        """Set the base model path."""
        self.send_keys(*self.BASE_MODEL_INPUT, model)

    def get_configs_dir(self):
        """Get the current configs directory value."""
        element = self.find_element(*self.CONFIGS_DIR_INPUT)
        return element.get_attribute("value")

    def get_model_name(self):
        """Get the current model name value."""
        element = self.find_element(*self.MODEL_NAME_INPUT)
        return element.get_attribute("value")

    def get_output_dir(self):
        """Get the current output directory value."""
        element = self.find_element(*self.OUTPUT_DIR_INPUT)
        return element.get_attribute("value")

    def get_base_model(self):
        """Get the current base model value."""
        element = self.find_element(*self.BASE_MODEL_INPUT)
        return element.get_attribute("value")

    def save_changes(self):
        """Save the changes in Basic Config tab."""
        self.click_element(*self.SAVE_BUTTON)
        self.wait_for_htmx()


class ModelConfigTab(BasePage):
    """Page object for Model Configuration tab."""

    MODEL_FAMILY_SELECT = (By.ID, "model_family")
    LORA_RANK_INPUT = (By.ID, "lora_rank")
    LORA_ALPHA_INPUT = (By.ID, "lora_alpha")

    def select_model_family(self, family):
        """Select model family."""
        from selenium.webdriver.support.select import Select

        select = Select(self.find_element(*self.MODEL_FAMILY_SELECT))
        select.select_by_value(family)

    def set_lora_rank(self, rank):
        """Set LoRA rank."""
        self.send_keys(*self.LORA_RANK_INPUT, str(rank))

    def set_lora_alpha(self, alpha):
        """Set LoRA alpha."""
        self.send_keys(*self.LORA_ALPHA_INPUT, str(alpha))


class TrainingConfigTab(BasePage):
    """Page object for Training Parameters tab."""

    LEARNING_RATE_INPUT = (By.ID, "learning_rate")
    BATCH_SIZE_INPUT = (By.ID, "train_batch_size")
    NUM_EPOCHS_INPUT = (By.ID, "num_train_epochs")
    MIXED_PRECISION_SELECT = (By.ID, "mixed_precision")

    def set_learning_rate(self, rate):
        """Set learning rate."""
        self.send_keys(*self.LEARNING_RATE_INPUT, str(rate))

    def set_batch_size(self, size):
        """Set batch size."""
        self.send_keys(*self.BATCH_SIZE_INPUT, str(size))

    def set_num_epochs(self, epochs):
        """Set number of epochs."""
        self.send_keys(*self.NUM_EPOCHS_INPUT, str(epochs))

    def select_mixed_precision(self, precision):
        """Select mixed precision mode."""
        from selenium.webdriver.support.select import Select

        select = Select(self.find_element(*self.MIXED_PRECISION_SELECT))
        select.select_by_value(precision)


class DatasetsTab(BasePage):
    """Page object for Datasets tab."""

    ADD_DATASET_BUTTON = (
        By.CSS_SELECTOR,
        r"""[x-on\:click="Alpine.store('datasets').showAddModal()"]""",
    )
    DATASET_CARDS = (By.CSS_SELECTOR, ".dataset-card")
    DATASET_DELETE_BUTTON = (By.CSS_SELECTOR, ".dataset-card .btn-danger")

    # Modal elements
    MODAL_NAME_INPUT = (By.ID, "modal-dataset-name")
    MODAL_PATH_INPUT = (By.ID, "modal-dataset-path")
    MODAL_SAVE_BUTTON = (By.CSS_SELECTOR, ".modal-footer .btn-primary")
    MODAL_CANCEL_BUTTON = (By.CSS_SELECTOR, ".modal-footer .btn-secondary")

    def add_dataset(self):
        """Click add dataset button."""
        self.click_element(*self.ADD_DATASET_BUTTON)
        # Wait for modal to appear
        self.wait.until(EC.visibility_of_element_located(self.MODAL_NAME_INPUT))

    def get_dataset_count(self):
        """Get the number of datasets."""
        return len(self.find_elements(*self.DATASET_CARDS))

    def fill_dataset_modal(self, name, path):
        """Fill the dataset modal form."""
        self.send_keys(*self.MODAL_NAME_INPUT, name)
        self.send_keys(*self.MODAL_PATH_INPUT, path)

    def save_dataset_modal(self):
        """Save the dataset modal."""
        self.click_element(*self.MODAL_SAVE_BUTTON)
        self.wait_for_htmx()

    def cancel_dataset_modal(self):
        """Cancel the dataset modal."""
        self.click_element(*self.MODAL_CANCEL_BUTTON)

    def delete_dataset(self, index=0):
        """Delete a dataset by index."""
        delete_buttons = self.find_elements(*self.DATASET_DELETE_BUTTON)
        if index < len(delete_buttons):
            delete_buttons[index].click()
            self.wait_for_htmx()
