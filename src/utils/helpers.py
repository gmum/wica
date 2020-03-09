def to_img(x, normalized):
    if normalized:
        x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x
