import enum

class TaskStatus(enum.Enum):
    INITIALIZING = "QTask is being initialized"
    QUEUED = "QTask is queued"
    VALIDATING = "QTask is being validated"
    RUNNING = "QTask is actively running"
    CANCELLED = "QTask has been cancelled"
    DONE = "QTask has successfully run"
    ERROR = "QTask incurred error"
