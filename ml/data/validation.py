"""Safe data errors shared by parsing and later task validation."""


class CsvValidationError(ValueError):
    """A stable error code and safe message, without paths or record values."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)
