from itertools import product

import numpy as np

def get_tiles(
    nols,
    nrows,
    width,
    height=None,
    step_w=None,
    step_h=None,
    col_offset=0,
    row_offset=0,
    cover_all=True,
):
    if step_w is None:
        step_w = width
    if height is None:
        height = width
    if step_h is None:
        step_h = step_w

    max_col_offset = int(np.ceil((nols - width) / step_w))
    # Remove all offsets such that offset+width > nols and add one offset to
    # reach nols
    col_offsets = list(range(col_offset, col_offset + nols, step_w))[
        : max_col_offset + int(cover_all)
    ]
    if cover_all:
        col_offsets[max_col_offset] = col_offset + nols - width

    max_row_offset = int(np.ceil((nrows - height) / step_h))
    # Remove all offsets such that offset+height > nols and add one offset to
    # reach nols
    row_offsets = list(range(row_offset, row_offset + nrows, step_h))[
        : max_row_offset + int(cover_all)
    ]
    if cover_all:
        row_offsets[max_row_offset] = row_offset + nrows - height

    offsets = product(col_offsets, row_offsets)
    for col_off, row_off in offsets:
        yield col_off, row_off, width, height


def main():
    for tile in get_tiles(1000, 1500, width=412, height=397, step_w=400, col_offset=10):
        print(tile)


if __name__ == "__main__":
    main()
