from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .models import FrameSampleResult


def format_timestamp(timestamp_sec: float) -> str:
    total_seconds = int(round(timestamp_sec))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_preview_contact_sheet(
    frame_samples: Sequence[FrameSampleResult],
    output_path: Path,
    *,
    title: str | None = None,
) -> Path | None:
    usable_samples = [sample for sample in frame_samples if Path(sample.frame_path).exists()]
    if not usable_samples:
        return None

    tile_width = 320
    tile_height = 200
    footer_height = 34
    padding = 18
    title_height = 42 if title else 0
    columns = min(4, max(1, len(usable_samples)))
    rows = math.ceil(len(usable_samples) / columns)
    canvas_width = columns * tile_width + (columns + 1) * padding
    canvas_height = rows * (tile_height + footer_height) + (rows + 1) * padding + title_height

    sheet = Image.new("RGB", (canvas_width, canvas_height), color=(248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    current_y = padding
    if title:
        draw.text((padding, current_y), title, fill=(20, 20, 20), font=font)
        current_y += title_height

    for index, sample in enumerate(usable_samples):
        column = index % columns
        row = index // columns
        tile_x = padding + column * (tile_width + padding)
        tile_y = current_y + row * (tile_height + footer_height + padding)
        frame_box = (tile_x, tile_y, tile_x + tile_width, tile_y + tile_height)

        with Image.open(sample.frame_path) as source_image:
            preview_image = ImageOps.contain(source_image.convert("RGB"), (tile_width, tile_height))
            offset_x = tile_x + (tile_width - preview_image.width) // 2
            offset_y = tile_y + (tile_height - preview_image.height) // 2
            sheet.paste(preview_image, (offset_x, offset_y))

        draw.rectangle(frame_box, outline=(188, 188, 188), width=1)
        footer_text = format_timestamp(sample.timestamp_sec)
        if sample.detected:
            footer_text += " | ELEPHANT"
            footer_color = (19, 113, 60)
        else:
            footer_text += " | empty"
            footer_color = (140, 64, 9)
        draw.text(
            (tile_x, tile_y + tile_height + 8),
            footer_text,
            fill=footer_color,
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90)
    return output_path

