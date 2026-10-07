"""Controlled error categories. `message` is safe to return to clients; `detail` is internal (logs only)."""


class RepairError(Exception):
    code = "internal_error"
    status = 500
    message = "Something went wrong."

    def __init__(self, message: str | None = None, detail: str = ""):
        super().__init__(detail or message or self.message)
        if message:
            self.message = message
        self.detail = detail


class EmptyImage(RepairError):
    code, status, message = "empty_image", 400, "No image data received."


class BadImage(RepairError):
    code, status, message = "bad_image", 400, "Could not read the uploaded file as an image."


class ImageTooLarge(RepairError):
    code, status, message = "image_too_large", 413, "Image is too large."


class ModelUnavailable(RepairError):
    code, status, message = "model_unavailable", 503, "The diagnostic model is not available right now."


class InferenceBusy(RepairError):
    code, status, message = "inference_busy", 503, "The server is busy. Please try again shortly."


class ModelTimeout(RepairError):
    code, status, message = "model_timeout", 504, "The model took too long to respond."


class ModelError(RepairError):
    code, status, message = "model_error", 502, "The model failed to process this image."


class InvalidModelOutput(RepairError):
    code, status, message = "invalid_model_output", 502, "The model returned an unusable answer. Please try again."
