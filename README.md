# sentinel2_download_keras_pipeline
Retrieves Sentinel-2 image from PEPS, extract only TCI, crop it by 64x64, introduces to EuroSAT sequential convolutional deep learning pre-trained model (by the used with "dnn_part.py" script) and obtain land use/cover classification result.

1-] Run "dnn_part" and save its pre-trained model to a working directory new folder.

2-] Use this directory throughout the "core" code. 

More explanations will be added, but bare minimum to run and obtain results via scripts are already there as comments. 
