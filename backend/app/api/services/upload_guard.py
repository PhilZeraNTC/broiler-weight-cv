"""Bounded async upload validation & Pillow decompression-bomb guard (D-02, T-04-01, T-04-02).

Strictly avoids OpenCV raw image decoders (PROIBIÇÃO 2): every upload passes
through Pillow header/magic verification, explicit 15 MiB byte capping, and a
40 MP decompression-bomb guard before any pixel array is materialized.
"""

from __future__ import annotations

import io

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024  # 15 MiB (D-02)
MAX_IMAGE_PIXELS_API: int = 40_000_000  # 40 MP hard ceiling (>5x Redmi 10's 7.51 MP)
ALLOWED_IMAGE_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP"})

_ERR_EMPTY_UPLOAD: str = (
    "Não foi possível decodificar esta imagem com segurança. "
    "Verifique se o arquivo não está corrompido e clique em 'Carregar Nova Foto'."
)
_ERR_OVERSIZE_UPLOAD: str = (
    "O arquivo ultrapassa o limite local de 15 MB. Reduza a resolução da foto "
    "ou selecione um preset da galeria para testar o pipeline."
)
_ERR_UNSUPPORTED_FORMAT: str = (
    "Formato de imagem não suportado. Envie um arquivo JPG, PNG ou WEBP "
    "ou escolha um dos presets auditados na galeria."
)
_ERR_CORRUPT_IMAGE: str = (
    "Não foi possível decodificar esta imagem com segurança. "
    "Verifique se o arquivo não está corrompido e clique em 'Carregar Nova Foto'."
)
_ERR_DECOMPRESSION_BOMB: str = (
    "Não foi possível decodificar esta imagem com segurança: dimensões inválidas "
    "ou acima do limite seguro de 40 MP. Clique em 'Carregar Nova Foto'."
)


async def read_and_validate_upload(upload: UploadFile) -> bytes:
    """Read bounded upload bytes asynchronously and validate image magic & pixel bounds."""
    try:
        raw = await upload.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await upload.close()

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail=_ERR_EMPTY_UPLOAD)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=_ERR_OVERSIZE_UPLOAD)

    await run_in_threadpool(validate_image_bytes, raw)
    return raw


def validate_image_bytes(raw: bytes) -> str:
    """Verify real Pillow format (JPEG/PNG/WEBP) and reject corrupt/decompression-bomb bytes."""
    if not raw:
        raise HTTPException(status_code=400, detail=_ERR_EMPTY_UPLOAD)

    try:
        with Image.open(io.BytesIO(raw)) as im:
            fmt = (im.format or "").upper()
            if fmt not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(status_code=415, detail=_ERR_UNSUPPORTED_FORMAT)
            w, h = im.size
            if w <= 0 or h <= 0 or (w * h) > MAX_IMAGE_PIXELS_API:
                raise HTTPException(status_code=400, detail=_ERR_DECOMPRESSION_BOMB)
            im.verify()
            return fmt
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=400, detail=_ERR_DECOMPRESSION_BOMB) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=_ERR_CORRUPT_IMAGE) from exc


__all__ = [
    "ALLOWED_IMAGE_FORMATS",
    "MAX_IMAGE_PIXELS_API",
    "MAX_UPLOAD_BYTES",
    "read_and_validate_upload",
    "validate_image_bytes",
]
