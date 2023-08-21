class StateTracker:
    # Class variables
    has_training_started = False

    @classmethod
    def start_training(cls):
        cls.has_training_started = True

    @classmethod
    def status_training(cls):
        return cls.has_training_started
