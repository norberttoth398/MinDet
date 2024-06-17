from MinDet.Process import mcmc, measure_funcs,  shape_calc, texture
import numpy as np

def test_mcmc():
    x = np.linspace(0,10,11)
    sample_m = np.array([0.45, 0.5, 0.55, 0.4, 0.6])
    sample_c = np.array([1,1.1,1.2,0.9,0.8])
    y_exp = x*sample_m.reshape(-1,1) + sample_c.reshape(-1,1)

    y_pred = mcmc.mcmc_predict(x, sample_m, sample_c, True)
    y_pred_red = mcmc.mcmc_predict(x, sample_m, sample_c, False)

    assert np.array_equal(y_pred, y_exp) == True and np.array_equal(y_pred_red[0], np.mean(y_exp, axis = 0)) and np.array_equal(y_pred_red[1], np.std(y_exp, axis = 0))
    