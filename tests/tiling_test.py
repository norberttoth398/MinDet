import MinDet

def test_tile_run():
    MinDet.tiling.tile_run("tests/labels/", (2,2), (180,180), 100, over_n=10)