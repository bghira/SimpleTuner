class StateTracker:
    # Class variables
    has_training_started = False
    calculate_luminance = False

    # Store the list of images, like a cache.
    all_image_files = {}
    all_caption_files = []

    @classmethod
    def start_training(cls):
        cls.has_training_started = True

    @classmethod
    def status_training(cls):
        return cls.has_training_started

    @classmethod
    def enable_luminance(cls):
        cls.calculate_luminance = True

    @classmethod
    def tracking_luminance(cls):
        return cls.calculate_luminance

    @classmethod
    def set_image_files(cls, image_files):
        cls.all_image_files = image_files

    @classmethod
    def get_image_files(cls):
        return cls.all_image_files

    @classmethod
    def has_image_files_loaded(cls):
        return len(cls.all_image_files) > 0

    @classmethod
    def set_caption_files(cls, caption_files):
        cls.all_caption_files = caption_files

    @classmethod
    def get_caption_files(cls):
        return cls.all_caption_files

    @classmethod
    def has_caption_files_loaded(cls):
        return len(cls.all_caption_files) > 0
