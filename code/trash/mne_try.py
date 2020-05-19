# Authors: Christopher Holdgraf <choldgraf@berkeley.edu>
#
# License: BSD (3-clause)
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from os import path as op

import mne
from mne.viz import ClickableImage  # noqa
from mne.viz import (plot_alignment, snapshot_brain_montage, set_3d_view)


print(__doc__)

subjects_dir = mne.datasets.sample.data_path() + '/subjects'
path_data = mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat'

# We've already clicked and exported
layout_path = op.join(op.dirname(mne.__file__), 'data', 'image')
layout_name = 'custom_layout.lout'


mat = loadmat(path_data)
ch_names = mat['ch_names'].tolist()
elec = mat['elec']  # electrode coordinates in meters
# Now we make a montage stating that the sEEG contacts are in head
# coordinate system (although they are in MRI). This is compensated
# by the fact that below we do not specicty a trans file so the Head<->MRI
# transform is the identity.
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, elec)),
                                        coord_frame='head')
info = mne.create_info(ch_names, 1000., 'ecog').set_montage(montage)
print('Created %s channel positions' % len(ch_names))


fig = plot_alignment(info, subject='sample', subjects_dir=subjects_dir,
                     surfaces=['pial'], meg=False)
set_3d_view(figure=fig, azimuth=200, elevation=70)
xy, im = snapshot_brain_montage(fig, montage)

# Convert from a dictionary to array to plot
xy_pts = np.vstack([xy[ch] for ch in info['ch_names']])

# Define an arbitrary "activity" pattern for viz
activity = np.linspace(100, 200, xy_pts.shape[0])

# This allows us to use matplotlib to create arbitrary 2d scatterplots
fig2, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im)
ax.scatter(*xy_pts.T, c=activity, s=200, cmap='coolwarm')
ax.set_axis_off()

fig2.savefig('./brain.png', bbox_inches='tight')  # For ClickableImage