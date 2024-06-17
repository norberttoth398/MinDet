import pandas as pd
from MinDet.Process import mcmc, measure_funcs,  shape_calc, texture
import numpy as np
def test_shape_calc():
    data = pd.read_csv("tests/cumulative_freq.csv")
    #data.head()
    d = data["1:1.00:1.00"]
    d2 = data["1:1.10:1.15"]
    d3 = data["1:17.0:20.0"]

    d_p = shape_calc.calc_zingg(d.to_numpy(), data)
    d2_p = shape_calc.calc_zingg(d2.to_numpy(), data)
    d3_p = shape_calc.calc_zingg(d3.to_numpy(), data)

    assert d_p[0][0] == 1 and d_p[0][1] == 1
    assert np.around(d_p[1][0], 3) == 0.063 and np.around(d_p[1][1], 3) == 0.077
    
    assert np.around(d2_p[0][0],2) == 0.91 and np.around(d2_p[0][1], 2) == 0.96
    assert np.around(d2_p[1][0], 3) == 0.060 and np.around(d2_p[1][1], 3) == 0.077

    assert np.around(d3_p[0][0],3) == 0.059 and np.around(d3_p[0][1], 2) == 0.85
    assert np.around(d3_p[1][0], 3) == 0.005 and np.around(d3_p[1][1], 3) == 0.081

    shape_calc.ZinggPlot(d, data)
    shape_calc.ZinggPlot(d, data, True)
