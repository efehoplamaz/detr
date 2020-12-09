import numpy as np
from . import wavfile
import warnings


def time_to_x_coords(time_in_file, sampling_rate, fft_win_length, fft_overlap):
    nfft = np.floor(fft_win_length*sampling_rate) # int() uses floor
    noverlap = np.floor(fft_overlap*nfft)
    return (time_in_file*sampling_rate-noverlap) / (nfft - noverlap)


# NOTE this is also defined in psot_process - not ideal
def x_coords_to_time(x_pos, sampling_rate, fft_win_length, fft_overlap):
    #nfft = np.floor(fft_win_length*sampling_rate)
    #noverlap = np.floor(fft_overlap*nfft)
    #return ((x_pos*(nfft - noverlap)) + noverlap) / sampling_rate
    return (1.0 - fft_overlap) * fft_win_length * (x_pos + 0.5)  # 0.5 is for center of temporal window


def generate_spectrogram(audio, sampling_rate, params, return_spec_for_viz=False, check_spec_size=True):
    max_freq = round(params['max_freq']*params['fft_win_length'])
    min_freq = round(params['min_freq']*params['fft_win_length'])

    # create spectrogram - numpy
    spec = gen_mag_spectrogram(audio, sampling_rate, params['fft_win_length'], params['fft_overlap'])

    if spec.shape[0] < max_freq:
        freq_pad = max_freq - spec.shape[0]
        spec = np.vstack((np.zeros((freq_pad, spec.shape[1]), dtype=spec.dtype), spec))
    spec_cropped = spec[-max_freq:spec.shape[0]-min_freq, :]

    if params['spec_scale'] == 'log':
        log_scaling = 2.0 * (1.0 / sampling_rate) * (1.0/(np.abs(np.hanning(int(params['fft_win_length']*sampling_rate)))**2).sum())
        spec = np.log1p(log_scaling*spec_cropped)
    elif params['spec_scale'] == 'pcen':
        # TODO need to clean up PCEN and remove dependencies on scipy and make params accessible
        spec = pcen(spec_cropped, b=0.005)
    elif params['spec_scale'] == 'none':
        pass

    if params['denoise_spec_avg']:
        spec = spec - np.mean(spec, 1)[:, np.newaxis]
        spec.clip(min=0, out=spec)

    # create spectrogram - pytorch
    # # TODO there are numerical diffrences in gen_mag_spectrogram_pytorch
    # audio = torch.from_numpy(audio).to(params['device'])
    # spec = gen_mag_spectrogram_pytorch(audio, sampling_rate, params['fft_win_length'], params['fft_overlap'])
    # if spec.shape[0] < max_freq:
    #     freq_pad = max_freq - spec.shape[0]
    #     spec = torch.cat((torch.zeros((freq_pad, spec.shape[1])).to(spec.device), spec), 0)
    # spec = spec[-max_freq:spec.shape[0]-min_freq, :]
    #
    # log_scaling = 2.0 * (1.0 / sampling_rate) * (1.0/(np.abs(np.hanning(int(params['fft_win_length']*sampling_rate)))**2).sum())
    # spec = torch.log(1.0 + log_scaling*spec)
    #
    # # denoise
    # spec = spec - spec.mean(1).unsqueeze(1)
    # torch.nn.functional.relu(spec, inplace=True)

    # needs to be divisible by specific factor - if not it should be padded
    if check_spec_size:
        assert((int(spec.shape[0]*params['resize_factor']) % params['spec_divide_factor']) == 0)
        assert((int(spec.shape[1]*params['resize_factor']) % params['spec_divide_factor']) == 0)

    # for visualization purposes - use log scaled spectrogram
    if return_spec_for_viz:
        log_scaling = 2.0 * (1.0 / sampling_rate) * (1.0/(np.abs(np.hanning(int(params['fft_win_length']*sampling_rate)))**2).sum())
        spec_for_viz = np.log1p(log_scaling*spec_cropped)
    else:
        spec_for_viz = spec

    return spec, spec_for_viz


def load_audio_file(audio_file, time_exp_fact, scale=False):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=wavfile.WavFileWarning)
        sampling_rate, audio_raw = wavfile.read(audio_file)

    if len(audio_raw.shape) > 1:
        raise Exception('Currently does not handle stereo files')
    sampling_rate = sampling_rate * time_exp_fact

    # scale and convert to float32
    if scale:
        audio_raw = audio_raw.astype(np.float32) / float(np.iinfo(audio_raw.dtype).max)
    else:
        audio_raw = audio_raw.astype(np.float32)

    return sampling_rate, audio_raw


def pad_audio(audio_raw, fs, ms, overlap_perc, resize_factor, divide_factor):
    # Adds zeros to the end of the raw data so that the generated sepctrogram
    # will be evenly divisible by `divide_factor`
    # Also deals with very short audio clips

    # TODO confrim this correct - simlar code in get_file_and_anns
    nfft = int(ms*fs)
    noverlap = int(overlap_perc*nfft)
    step = nfft - noverlap
    min_size = int(divide_factor*(1.0/resize_factor))
    spec_width = ((audio_raw.shape[0]-noverlap)//step) * resize_factor

    if spec_width < min_size or (np.floor(spec_width) % divide_factor) != 0:
        # need to be at least min_size
        div_amt = np.ceil(spec_width / float(divide_factor))
        div_amt = np.maximum(1, div_amt)
        target_size = int(div_amt*divide_factor*(1.0/resize_factor))

        diff = target_size*step + noverlap - audio_raw.shape[0]
        audio_raw = np.hstack((audio_raw, np.zeros(diff, dtype=audio_raw.dtype)))

    return audio_raw


def gen_mag_spectrogram(x, fs, ms, overlap_perc):
    # Computes magnitude spectrogram by specifying time.

    x = x.astype(np.float32)
    nfft = int(ms*fs)
    noverlap = int(overlap_perc*nfft)

    # window data
    step = nfft - noverlap
    shape = (nfft, (x.shape[-1]-noverlap)//step)
    strides = (x.strides[0], step*x.strides[0])
    x_wins = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # apply window
    x_wins = np.hanning(x_wins.shape[0]).astype(np.float32)[..., np.newaxis] * x_wins

    # do fft
    # note this will be much slower if x_wins.shape[0] is not a power of 2
    complex_spec = np.fft.rfft(x_wins, axis=0)

    # calculate magnitude
    #spec = (np.conjugate(complex_spec) * complex_spec).real
    # same as:
    spec = np.absolute(complex_spec)**2

    # remove DC component and flip vertical orientation
    spec = np.flipud(spec[1:, :])

    return spec


from scipy.signal import lfilter
def pcen(S, gain=0.98, bias=2, power=0.5, time_constant=0.400, eps=1e-6, b=0.5, axis=-1):

    # if b is None:
    #     t_frames = time_constant * sr / float(hop_length)
    #     # By default, this solves the equation for b:
    #     #   b**2  + (1 - b) / t_frames  - 2 = 0
    #     # which approximates the full-width half-max of the
    #     # squared frequency response of the IIR low-pass filter
    #
    #     b = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)

    # Temporal smoothing
    S_smooth, _ = lfilter([b], [1, b - 1], S, zi=np.ones((1,1))*(1-b), axis=-1)

    # Adaptive gain control
    # Working in log-space gives us some stability, and a slight speedup
    smooth = np.exp(-gain * (np.log(eps) + np.log1p(S_smooth / eps)))

    # Dynamic range compression
    if power == 0:
        S_out = np.log1p(S*smooth)
    elif bias == 0:
        S_out = np.exp(power * (np.log(S) + np.log(smooth)))
    else:
        S_out = (bias**power) * np.expm1(power * np.log1p(S*smooth/bias))

    return S_out.astype(np.float32)
