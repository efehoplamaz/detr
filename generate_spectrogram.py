"""
Script to generate spectrogram. 
"""
import utils.audio_utils as au
import utils.visualize as viz

import matplotlib.pyplot as plt
import numpy as np
import json
import os


def get_spectrogram_sampling_rate(audio_file):

	# specify the file of interest
	time_expansion_factor = 1.0

	# fft parameters
	params = {}
	params['fft_win_length'] = 1024 / 441000.0  # 1024 / 441000.0
	params['resize_factor'] = 0.5     # resize so the spectrogram at the input of the network
	params['fft_overlap']    = 0.75  # 0.75
	params['max_freq'] = 120000 # will result in bin id 279 if fft_win_length == 1024 / 441000.0
	params['min_freq'] = 10000  # will result in bin id 23
	params['denoise_spec_avg'] = True  # removes the mean for each frequency band
	params['scale_raw_audio'] = False  # scales the raw audio by min max of dtype
	params['spec_scale'] = 'pcen'       # 'log', 'pcen', 'none'

	# read the audio file 
	sampling_rate, audio = au.load_audio_file(audio_file, time_expansion_factor)
	duration = audio.shape[0] / sampling_rate
	#print('File duration: {} seconds'.format(duration))

	# generate spectrogram for visualization
	_, spec = au.generate_spectrogram(audio, sampling_rate, params, True, False)

	return spec, sampling_rate


def display_spectrogram(audio_file, spec, sampling_rate, annotations):

	# fft parameters
	params = {}
	params['fft_win_length'] = 1024 / 441000.0  # 1024 / 441000.0
	params['resize_factor'] = 0.5     # resize so the spectrogram at the input of the network
	params['fft_overlap']    = 0.75  # 0.75
	params['max_freq'] = 120000 # will result in bin id 279 if fft_win_length == 1024 / 441000.0
	params['min_freq'] = 10000  # will result in bin id 23
	params['denoise_spec_avg'] = True  # removes the mean for each frequency band
	params['scale_raw_audio'] = False  # scales the raw audio by min max of dtype
	params['spec_scale'] = 'pcen'       # 'log', 'pcen', 'none'


	# load annotations
	anns = annotations

	# extract out the boxes - also need to add an extra field for visualization
	gt = [ann for ann in anns['annotation']]
	for gg in gt:
	    gg['det_prob'] = 1.0
	    
	# display the annotations on top of the spectrogram
	start_time = 0.0
	fig = plt.figure(1, figsize=(spec.shape[1]/100, spec.shape[0]/100), dpi=100, frameon=False)
	spec_duration = au.x_coords_to_time(spec.shape[1], sampling_rate, params['fft_win_length'], params['fft_overlap'])
	viz.create_box_image(spec, fig, gt, start_time, start_time+spec_duration, spec_duration, params, spec.max()*1.1, False)
	plt.ylabel('Freq - kHz')
	plt.xlabel('Time - secs')
	plt.title(os.path.basename(audio_file))
	plt.show()