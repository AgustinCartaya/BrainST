"""Piano-keyboard and segmentation-highlighting rendering utilities.

Used by the "Brain Concert" bonus demo to draw a virtual piano keyboard
(optionally with brain-slice thumbnails embedded above each white key) and
to highlight a specific anatomical structure within a 2D segmentation
slice.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WHITE_KEYS = ["C", "D", "E", "F", "G", "A", "B"]

# Maps each black-key note name to the index of the white key immediately
# to its left (0-based position within WHITE_KEYS), used to position the
# black key between two white keys.
BLACK_KEYS = {
    "C#": 0,
    "D#": 1,
    "F#": 3,
    "G#": 4,
    "A#": 5,
}


def draw_piano(
    active_note: str | None = None,
    width: int = 1400,
    height: int = 300,
    active_color: str = "red",
    font_size: int = 48,
    white_key_images: list[np.ndarray | Image.Image] | None = None,
    white_key_text_y: int = 220,
    black_key_text_y: int = 120,
    white_key_image_y: int = 10,
    white_image_height: int = 140,
    white_key_image_x_offset: int | None = None,
) -> np.ndarray:
    """Render a one-octave piano keyboard, optionally with thumbnails over each white key.

    Args:
        active_note: Note name (e.g. ``"C"`` or ``"C#"``) to highlight with
            ``active_color``, or ``None`` for no highlighted key.
        width: Total image width, in pixels.
        height: Height of the white-key area, in pixels.
        active_color: Fill color used for the highlighted key.
        font_size: Font size for the note-name labels.
        white_key_images: Optional list of images (NumPy arrays or PIL
            Images), one per white key in ``WHITE_KEYS`` order, drawn above
            (or overlapping) each white key. Entries may be ``None`` to
            skip a given key.
        white_key_text_y: Y-coordinate (relative to the piano's own
            coordinate space, before any canvas offset) where the white-key
            note labels are drawn.
        black_key_text_y: Y-coordinate where the black-key note labels are
            drawn.
        white_key_image_y: Y-coordinate where ``white_key_images`` are
            pasted. May be negative to place images above the piano itself,
            in which case the canvas is automatically extended upward.
        white_image_height: Maximum height (pixels) each pasted image is
            resized to, preserving aspect ratio.
        white_key_image_x_offset: Fixed horizontal offset (from the left
            edge of each key) used to position each image, or ``None`` to
            center the image within its key.

    Returns:
        An RGB image (as a ``uint8`` NumPy array) of the rendered keyboard,
        with its height automatically extended to accommodate images that
        are placed outside the piano's own ``[0, height]`` vertical range.
    """
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)

    # If white_key_image_y is negative, images extend above the piano; if
    # white_key_image_y + white_image_height exceeds `height`, images
    # extend below it. Both cases require growing the canvas so nothing
    # gets clipped.
    top_offset = max(0, -white_key_image_y)
    bottom_extra = max(0, white_key_image_y + white_image_height - height)
    canvas_height = height + top_offset + bottom_extra

    img = Image.new("RGB", (width, canvas_height), "white")
    draw = ImageDraw.Draw(img)

    key_width = width // len(WHITE_KEYS)

    # Everything drawn for the piano itself is shifted down by top_offset
    # so that content placed above the piano (negative white_key_image_y)
    # remains within the canvas.
    piano_y_offset = top_offset
    text_y = white_key_text_y + piano_y_offset

    # ---- White keys ----
    for i, note in enumerate(WHITE_KEYS):
        x0 = i * key_width
        x1 = x0 + key_width

        color = active_color if note == active_note else "white"

        draw.rectangle(
            [x0, piano_y_offset, x1, height + piano_y_offset],
            fill=color,
            outline="black",
            width=3,
        )

        # Paste the associated thumbnail image, if provided.
        if white_key_images is not None and i < len(white_key_images) and white_key_images[i] is not None:
            key_image = white_key_images[i]

            if isinstance(key_image, np.ndarray):
                key_image = Image.fromarray(key_image)

            # Work on a copy: `.thumbnail()` mutates in place.
            key_image = key_image.copy()
            key_image.thumbnail((key_width, white_image_height), Image.Resampling.LANCZOS)

            if white_key_image_x_offset is None:
                paste_x = x0 + (key_width - key_image.width) // 2
            else:
                paste_x = x0 + white_key_image_x_offset
            paste_y = white_key_image_y + piano_y_offset

            img.paste(key_image, (paste_x, paste_y))

        # Draw the note label, centered horizontally within the key.
        bbox = draw.textbbox((0, 0), note, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (x0 + (key_width - text_width) / 2, text_y),
            note,
            fill="black",
            font=font,
        )

    # ---- Black keys ----
    black_width = key_width * 0.6
    black_height = 180

    for note, left_white_key_index in BLACK_KEYS.items():
        x = (left_white_key_index + 1) * key_width - black_width / 2

        color = active_color if note == active_note else "black"

        draw.rectangle(
            [x, piano_y_offset, x + black_width, black_height + piano_y_offset],
            fill=color,
        )

        bbox = draw.textbbox((0, 0), note, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (x + (black_width - text_width) / 2, black_key_text_y + piano_y_offset),
            note,
            fill="white",
            font=font,
        )

    return np.array(img)


def get_highlighted_structure(
    segmentation_2d: np.ndarray,
    highlighted_indices: list[int],
    bg_color: tuple[int, int, int] = (0, 0, 0),
    brain_color: tuple[int, int, int] = (128, 128, 128),
    highlighted_color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Convert a 2D segmentation slice into an RGB image highlighting selected labels.

    Args:
        segmentation_2d: 2D integer label map.
        highlighted_indices: Label values to render in ``highlighted_color``.
        bg_color: RGB color used for background voxels (label ``0``).
        brain_color: RGB color used for all other (non-background,
            non-highlighted) segmentation voxels.
        highlighted_color: RGB color used for voxels whose label is in
            ``highlighted_indices``.

    Returns:
        An RGB image (``uint8`` array of shape ``segmentation_2d.shape +
        (3,)``) with the requested structure(s) highlighted.
    """
    rgb_image = np.zeros((*segmentation_2d.shape, 3), dtype=np.uint8)
    rgb_image[:] = bg_color

    # Color every non-background voxel with the generic "brain" color first...
    rgb_image[segmentation_2d > 0] = brain_color

    # ...then overwrite the requested structure(s) with the highlight color.
    for label in highlighted_indices:
        rgb_image[segmentation_2d == label] = highlighted_color

    return rgb_image