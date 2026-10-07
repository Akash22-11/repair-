"""Structured key=value logging with a per-request id. Never pass images, prompts or secrets in here."""
import contextvars
import logging

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_log = logging.getLogger("repair")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False


def event(name: str, level: int = logging.INFO, exc_info=False, **kv) -> None:
    kv = {"request_id": request_id_var.get(), **kv}
    _log.log(level, "%s %s", name, " ".join(f"{k}={v}" for k, v in kv.items()), exc_info=exc_info)
