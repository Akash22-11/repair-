"""Image validation + preprocessing. Nothing here trusts filename, MIME type or client-supplied sizes."""
import io
import warnings

from PIL import Image, ImageOps

from . import config
from .errors import BadImage, ImageTooLarge

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS  # Pillow's own bomb guard (errors at 2x this)
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF"}
GRID = 32            # Qwen3-VL: 16px patches x 2x2 merge -> 1 image token per 32x32 px
MIN_SIDE = 32
MAX_ASPECT = 20


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


def prepare(raw: bytes):
    """-> (PIL RGB image ready for the model, (w, h) the model will see, (w, h) of the original)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(raw))
            fmt, (w, h) = probe.format, probe.size
            if fmt not in ALLOWED_FORMATS:
                raise BadImage("Unsupported image format. Use JPEG, PNG or WEBP.")
            if w * h > config.MAX_IMAGE_PIXELS:
                raise ImageTooLarge("Image resolution is too high.")
            if min(w, h) < MIN_SIDE or max(w, h) / min(w, h) > MAX_ASPECT:
                raise BadImage("Image dimensions are not usable.")
            probe.verify()                              # structural check; invalidates `probe`
            img = Image.open(io.BytesIO(raw))
            img.load()                                  # full decode: catches truncated/corrupt data
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise ImageTooLarge("Image resolution is too high.", detail=type(e).__name__) from e
    except (BadImage, ImageTooLarge):
        raise
    except Exception as e:
        raise BadImage(detail=f"{type(e).__name__}") from e

    img = _to_rgb(ImageOps.exif_transpose(img))
    ow, oh = img.size
    s = min(1.0, config.IMAGE_MAX_SIDE / max(ow, oh))
    nw = max(GRID, round(ow * s / GRID) * GRID)
    nh = max(GRID, round(oh * s / GRID) * GRID)
    if (nw, nh) != (ow, oh):
        img = img.resize((nw, nh), Image.LANCZOS)
    return img, (nw, nh), (ow, oh)
