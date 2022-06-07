def LOS4(map, location):
    """
    all cells can be seen from location including it    """
    h, w = len(map), len(map[0])
    x, y = decode(location, w)
    can_see = set()
    can_see.add(location)
    assert map[x, y] == 0, "illegal position!"
    left = right = up = down = True
    times = 1
    while any([left, right, up, down]):
        left_border, right_border = y - times, y + times
        up_border, down_border = x - times, x + times
        if left and left_border >= 0 and map[x][left_border] == 0:
            can_see.add(encode(x, left_border, w))
        else:
            left = False
        if right and right_border < w and map[x][right_border] == 0:
            can_see.add(encode(x, right_border, w))
        else:
            right = False
        if up and up_border >= 0 and map[up_border][y] == 0:
            can_see.add(encode(up_border, y, w))
        else:
            up = False
        if down and down_border < h and map[down_border][y] == 0:
            can_see.add(encode(down_border, y, w))
        else:
            down = False
        times += 1
    return can_see


def LOS8(map, location):
    """
    including diagonal line compared with LOS4
    """
    h, w = len(map), len(map[0])
    x, y = decode(location, w)
    can_see = set()
    can_see.add(location)
    assert map[x, y] == 0, "illegal position!"
    left = right = up = down = True
    upleft = downleft = upright = downright = True
    times = 1
    while any([left, right, up, down, upleft, downleft, upright, downright]):
        left_border, right_border = y - times, y + times
        up_border, down_border = x - times, x + times
        if left and left_border >= 0 and map[x][left_border] == 0:
            can_see.add(encode(x, left_border, w))
        else:
            left = False
        if right and right_border < w and map[x][right_border] == 0:
            can_see.add(encode(x, right_border, w))
        else:
            right = False
        if up and up_border >= 0 and map[up_border][y] == 0:
            can_see.add(encode(up_border, y, w))
        else:
            up = False
        if down and down_border < h and map[down_border][y] == 0:
            can_see.add(encode(down_border, y, w))
        else:
            down = False
        if upleft and left_border >= 0 and up_border >= 0 and \
                map[up_border][left_border] == 0:
            can_see.add(encode(up_border, left_border, w))
        else:
            upleft = False
        if downleft and left_border >= 0 and down_border < h and \
                map[down_border][left_border] == 0:
            can_see.add(encode(down_border, left_border, w))
        else:
            downleft = False
        if upright and right_border < w and up_border >= 0 and \
                map[up_border][right_border] == 0:
            can_see.add(encode(up_border, right_border, w))
        else:
            upright = False
        if downright and right_border < w and down_border < h and \
                map[down_border][right_border] == 0:
            can_see.add(encode(down_border, right_border, w))
        else:
            downright = False
        times += 1
    return can_see


def encode(x, y, w):
    return x * w + y


def decode(code, w):
    return code // w, code % w