import opc

def test_hex():
    assert opc.hex_to_rgb("#0000ff") == opc.RGB(0, 0, 255)
    assert opc.hex_to_rgb("#0300ff") == opc.RGB(3, 0, 255)
    assert opc.hex_to_rgb("#0003ff") == opc.RGB(0, 3, 255)
    assert opc.hex_to_rgb("#ffFFff") == opc.RGB(255, 255, 255)

