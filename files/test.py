import lights
import opc

def test_hex():
    assert opc.hex_to_rgb("#0000ff") == opc.RGB(0, 0, 255)
    assert opc.hex_to_rgb("#0300ff") == opc.RGB(3, 0, 255)
    assert opc.hex_to_rgb("#0003ff") == opc.RGB(0, 3, 255)
    assert opc.hex_to_rgb("#ffFFff") == opc.RGB(255, 255, 255)

def test_conversion():
    for rgb in [
        (1, 54, 6),
        (100, 5, 61),
        (222, 54, 6),
        (99, 14, 255),
        (120, 44, 22),
        (1, 154, 120)
    ]:
        r_, g_, b_ = lights.hsv_to_rgb(lights.rgb_to_hsv(rgb))
        rounded = (int(round(r_)), int(round(g_)), int(round(b_)))
        assert rgb == rounded

    assert lights.rgb_to_hsv((1, 3, 4)) == (200.0, 0.75, 4)
