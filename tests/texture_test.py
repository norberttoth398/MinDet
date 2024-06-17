from MinDet.Process import mcmc, measure_funcs,  shape_calc, texture
import numpy as np

def test_texture():
    sizes = np.random.randint(1, 1000, 300)
    aspects = np.random.rand(300)*5 + 0.5

    bin_edges = np.linspace(0.001,2000, 100)

    _ = texture.gen_aspect_ratio_data(sizes, aspects, bin_edges)

    _ = texture.CSD_plot(sizes)
    _ = texture.CSD_plot(sizes, geometric = True, min_max= (0.1, 1000))

    _ = texture.AR_plot(aspects, sizes)
    _ = texture.AR_plot(aspects, sizes, geometric=True, min_max= (0.1, 1000))