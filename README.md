# 2020 XWBank AI Competition - Top 10

The purpose of this contest is to predict behaviors (playing games, watching videos, browsing webs...) under different scenes (stadning, sitting, lying...) based on the acceleration sensor data, including gravity acceleration (ACC_Xg, ACC_Yg, ACC_Zg) and non-gravity acceleration (ACC_x, ACC_Y, ACC_z) of the mobile phone, in order to detect financial fraud operated on mobile phones.

The Competition Page <https://www.kesci.com/home/competition/5ece30cc73a1b3002c9f1bf5/content/0>

## Summary

+ Resample (key, challenging)
	+ FFT upsampling
	+ randomly sampling from original sequence and concatenating

+ Feature Engineering
	+ mod of acc and acc_g
	+ angles
	
+ Data Augmentation
	+ jitter
	+ rotation
	+ permutation
	+ ...
	
+ Models
	+ Temporal CNN
	+ CNN 2D
	+ LSTM FCN
	+ LSTM

+ Ensemble
	+ Blending with coefs based on local CV