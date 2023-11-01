# Example Inference

## Batch inference

Below is a python script used for batch inference of samples from Skuggafjoll on a distributed cluster computer. This script takes JPEG images of the given names from separate directories of each sample name. Config and checkpoint files are defined and passed to the function. We perform inference using CPU only - though if available GPU would be faster. Tile images are produced 1000px by 1000px with 250px overlap.

	from PlagDetect import run
	import numpy as np
	import matplotlib.pyplot as plt
	samples = ["HOR11_01A", "HOR11_01C", "HOR11_02A", "HOR11_02B", "HOR12_03A", "HOR12_03B", "SKU11_02B", "SKU11_03",
        	"SKU12_09B", "SKU12_15A", "SKU12_15B", "SKU12_16", "SKU12_26", "SKU12_25", "SKU12_20","SKU12_22", "SKU12_09A",
         	"SKU12_15C","SKU11_02A_1", "SKU11_02A_2"]

	config = "models/mmdet_detectoRS_dataAug.py"
	checkpoint = "models/27May_r50_detectoRS_extraAug.pth"
	for item in samples:
    	run.__run__(item + ".jpg", item, config, checkpoint, "cpu", 1000,1000, 250)



## Tile_only

Below is a similar script for just tiling to be performed on pre-existing results - this is not necessary unless detection score threshold or NMS threshold is to be changed. Note that tile images are assumed to be square and all results will be saved in the "tile_only" directory in this case.

	from PlagDetect import run
        import numpy as np
        import matplotlib.pyplot as plt
        samples = ["HOR11_01A", "HOR11_01C", "HOR11_02A", "HOR11_02B", "HOR12_03A", "HOR12_03B", "SKU11_02B", "SKU11_03",
                "SKU12_09B", "SKU12_15A", "SKU12_15B", "SKU12_16", "SKU12_26", "SKU12_25", "SKU12_20","SKU12_22", "SKU12_09A",
                "SKU12_15C","SKU11_02A_1", "SKU11_02A_2"]

	for name in samples:
    		img_path = name + "/"
		run.__tile_only__(name, img_path,img_path +"labels", "tile_only/", 1000,250, thresh = 0.8)



