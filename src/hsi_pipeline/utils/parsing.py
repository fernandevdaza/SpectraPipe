def parse_pixels_inline(pixels_str: str) -> list[tuple[int, int]]:
    """Parse '120,80;125,85' into [(120,80), (125,85)]."""
    result = []
    for pixel in pixels_str.split(";"):
        if not pixel.strip():
            continue
        parts = pixel.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid pixel format: '{pixel}'. Expected 'x,y'")
        try:
            x, y = int(parts[0].strip()), int(parts[1].strip())
            result.append((x, y))
        except ValueError:
             raise ValueError(f"Invalid coordinates: '{pixel}'. Must be integers.")
    return result
