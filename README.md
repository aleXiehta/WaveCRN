# WaveCRN
WaveCRN: An Efficient Convolutional Recurrent Neural Network for End-to-end Speech Enhancement

This repo is an example usage of the proposed model. ```example/main.ipynb``` shows a minimal SE pipline and a visulization of the enhanced speech sample.

## Model Architecture
![image](https://github.com/aleXiehta/WaveCRN/blob/master/images/model.png)
The architecture of the proposed WaveCRN model. For local feature extraction, a 1D CNN maps the noisy audio **x** into a 2D feature map **F**. Bi-SRU then encodes **F** into an restricted  feature mask (RFM) **M**, which is element-wisely multiplied by **F** to generate a masked feature map **F'**. Finally, a transposed 1D convolution layer recovers the enhanced waveform **y** from **F'**.

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

## Q&A
**Q**:Table 1 shows that the proposed method, WaveCRN, performs slightly better than Wave-U-Net but without confidence intervals it is not possible to ascertain whether these differences are statistically significant or just happened by chance. 

**A**: We would like to emphasize that the major advantage of WaveCRN is that it can achieve comparable performance to state-of-the-art SE methods (such as Wave-U-Net [36]) while requiring much less model complexity and computational costs. In the revised paper, we have clearly highlighted the main advantages of WaveCRN: (a) In Section III-B, we reported “It can be clearly seen from Table I that WaveCRN outperforms other models in terms of all perceptual and signal-level evaluation metrics.” (b) From Tables I and II, It is clear that WaveCRN uses a smaller size and less computational costs and achieves comparable performance to WaveCBLSTM. Table II shows that WaveCRN is 11.48/5.56 times faster than Wave-U-Net in forward/back-propagation pass in the training stage, while having only 27% of the parameter number as compared to Wave-U-Net. Due to the space limitation, we did not report the model size and the computational cost in Table II. Instead, we report Table II.R1 in our GitHub page (https://github.com/aleXiehta/WaveCRN). If you advise us to include the result, we will use Table II.R1 to replace Table II in the current manuscript.
<img src="https://github.com/aleXiehta/WaveCRN/blob/master/images/table2.png" />
