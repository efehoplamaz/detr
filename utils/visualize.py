import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
from matplotlib import patches
from matplotlib.collections import PatchCollection


def plot_bounding_box_patch(pred_nms, freq_scale, start_time):
    patch_collect = []
    for bb in range(len(pred_nms)):
        xx = pred_nms[bb]['start_time'] - start_time
        ww = pred_nms[bb]['end_time'] - pred_nms[bb]['start_time']
        yy = pred_nms[bb]['low_freq'] / freq_scale
        hh = (pred_nms[bb]['high_freq'] - pred_nms[bb]['low_freq']) / freq_scale
        patch_collect.append(patches.Rectangle((xx,yy),ww,hh, linewidth=1, edgecolor='w',
                                 facecolor='none', alpha=pred_nms[bb]['det_prob']))
    return patch_collect


def create_box_image(spec, fig, detections_ip, start_time, end_time, duration, params, max_val, hide_axis=True):
    # filter detections
    detections = []
    for bb in detections_ip:
        if (bb['start_time'] >= start_time) and (bb['end_time'] < end_time):
            detections.append(bb)

    # create figure
    freq_scale = 1000  # turn Hz in kHz
    y_extent = [0, duration, params['min_freq']//freq_scale, params['max_freq']//freq_scale]

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    if hide_axis:
        ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(spec, aspect='auto', cmap='plasma', extent=y_extent, vmin=0, vmax=max_val)
    boxes = plot_bounding_box_patch(detections, freq_scale, start_time)
    ax.add_collection(PatchCollection(boxes, match_original=True))
    plt.grid(False)
