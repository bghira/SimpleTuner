class StateTracker:
    # Class variables
    has_training_started = False
    calculate_luminance = False
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