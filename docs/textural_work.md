# Textural Work

There is custom code written for textural analysis of segmentation results that may be found in the [Process](./references/measure.md) set of reference pages. Available tools include [measure](./references/measure.md), [texture](./references/texture.md), [ShapeCalc](./references/shape_calc.md) and [MCMC prediction](./references/mcmc.md).

## Measure

Size and aspect ratio data are calculated for each separate instance mask in the label image passed to the function "gen_texture_data" with choice of measurement type for size - "area^(0.5)" or "length", as well as measurement technique - "best fit ellipse" or "bounding box". Default choice is "area^(0.5)" and "best fit ellipse".

	size, aspect_ratio = gen_texture_data(segmentation_output, sqrt_area = True, BBOX = False)

## Texture

Here a set of functions are available to generate conventional textural plots such as crystal size distributions and aspect ratio distributions as a function of crystal size. Data binning is performed automatically based on the number of bins requested, with availabilit of both geometric and linear binning strategies. Manual min/max size values can be set by using the "min_max" and "manual_lims" options. Matplotlib axis object can be passed to the function to be incorporated manually into larger figures.

	 AR_plot(aspect_ratio, size, auto_bins=True, bins=10, geometric=False, min_max=None, manual_lims=False, x_lims=None, y_lims=None, ax=None)

	 CSD_plot(size, auto_bins=True, bins=10, geometric=False, min_max=None, manual_lims=False, x_lims=None, y_lims=None, ax=None) 

## ShapeCalc

We implemented the ShapeCalc algorithm from [Mangler et. al (2022)](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj34ZWpqIv_AhXQjFwKHYFEBEQQFnoECAwQAQ&url=https%3A%2F%2Flink.springer.com%2Farticle%2F10.1007%2Fs00410-022-01922-9&usg=AOvVaw3wfC685AWyQi3QNzeKJXAt) to create zingg plots and interrogate the distribution of likely 3D crystal shapes. "Feature" refers to a list of the observed cumulative 1/ar distributions, "models" refers to the slicing model results used for the analysis. Users have the option to show either the mean of the shape distributions as the most likely shape or the best matching model. "n" refers to the number of best-fit models plotted and it's possible to pass a custom matplotlib axis object to this function as well. Custom markers for each separate set of cumulative distributions are supported. 

	 ZinggPlot(feature, models, mean=False, n=30, ax=None, marker=None) 

## MCMC Prediction

We make it simple and easy to run MCMC prediction over a set of aspect ratio values from the [crystal aspect ratio and crystallisation time](./examples/Sill_calibration.ipynb) fit shown in the examples. Two functions are written depending on the way the MCMC results are used by the user - either by passing the "m" and "c" samples directly or from the saved MCMC results file. The "return_full" option allows the user to probe the full distribution of results, otherwise only the mean and standard deviation of the result distribution are returned. By default, "return_full" is set to False.

	 mcmc_predict(x, sample_m, sample_c, return_full=False) 

	 mcmc_predict_from_file(x, file, return_full=False) 
