# WaveCRN
WaveCRN: An Efficient Convolutional Recurrent Neural Network for End-to-end Speech Enhancement

This repo is an example usage of the proposed model. ```example/main.ipynb``` shows a minimal SE pipline and a visulization of the enhanced speech sample.

## Model Architecture
![image](https://github.com/aleXiehta/WaveCRN/blob/master/images/model.png)
The architecture of the proposed WaveCRN model. For local feature extraction, a 1D CNN maps the noisy audio \textbf{x} into a 2D feature map **F**. Bi-SRU then encodes **F** into an restricted  feature mask (RFM) **M**, which is element-wisely multiplied by **F** to generate a masked feature map **F'**. Finally, a transposed 1D convolution layer recovers the enhanced waveform **y** from **F'**.

## Experimental Results
### Results of Voice Bank + Demand Dataset
<!--<img src="https://github.com/aleXiehta/WaveCRN/blob/master/images/denoise.png" />-->
<!--<img src="https://github.com/aleXiehta/WaveCRN/blob/master/images/denoise_spec.png" />-->
<img src="https://github.com/aleXiehta/WaveCRN/blob/master/images/denoise_spec_metric.png" />

### Results of TIMIT for Compressed Speech Restoration
<img src="https://github.com/aleXiehta/WaveCRN/blob/master/images/restoration.png" />

## Requirements
torch==1.4.0<br/>
sru==2.3.5
