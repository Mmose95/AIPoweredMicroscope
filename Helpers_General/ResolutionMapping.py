def map_low_to_high_res(x_low, y_low, w_low, h_low,
                          low_res_shape=(8192, 5641),
                          high_res_shape=(86306, 59435),
                          tile_size=(2584, 1936)):
    """
    Maps a selected region from the low-resolution image to the high-resolution image.

    Parameters:
    - x_low, y_low: Top-left corner coordinates of the selected region in the low-res image.
    - w_low, h_low: Width and height of the selected region in the low-res image.
    - low_res_shape: (width, height) of the low-resolution image.
    - high_res_shape: (width, height) of the high-resolution image.
    - tile_size: Desired size of regions in the high-res image (default 2584x1936).

    Returns:
    - (x_high, y_high, w_high, h_high): Mapped region in the high-resolution image.
    """
    scale_x = high_res_shape[0] / low_res_shape[0]
    scale_y = high_res_shape[1] / low_res_shape[1]

    # Scale the coordinates and size
    x_high = int(x_low * scale_x)
    y_high = int(y_low * scale_y)
    w_high = int(w_low * scale_x)
    h_high = int(h_low * scale_y)

    # Adjust to match the tile size
    w_high = (w_high // tile_size[0]) * tile_size[0]
    h_high = (h_high // tile_size[1]) * tile_size[1]

    return x_high, y_high, w_high, h_high


# Example usage:
# Select a region (x=100, y=150) with width=50 and height=40 in the low-res image
x_high, y_high, w_high, h_high = map_low_to_high_res(100, 150, 50, 40)
print(f"Mapped high-res region: (x={x_high}, y={y_high}, width={w_high}, height={h_high})")