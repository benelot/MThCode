import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import os

df0 = pd.read_pickle('../data/performance_SLP.pkl')
df1 = pd.read_pickle('../data/performance_SRNN_L00.pkl')
df2 = pd.read_pickle('../data/performance_SRNN_L50.pkl')

df = pd.DataFrame()
df = df.append(df0, ignore_index=True)
df = df.append(df1, ignore_index=True)
df = df.append(df2, ignore_index=True)

plt.figure(figsize=(3, 1.4))
sns.set_style('ticks')
ax = sns.boxplot(x='correlation', y='model', data=df, fliersize=0, color='grey', order=['SLP', 'SRNN_L50', 'SRNN_L00'])
plt.xlabel('Correlation [-]')
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_ylabel('')
ax.set_yticklabels([])
ax.set_xlim(1, .7)
ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
c_edge = 'black'
for k, artist in enumerate(ax.artists):
    artist.set_edgecolor(c_edge)
    for l in range(k * 6, k * 6 + 6):
        ax.lines[l].set_color(c_edge)
        ax.lines[l].set_mfc(c_edge)
        ax.lines[l].set_mec(c_edge)

plt.tight_layout()
plt.savefig('figures/model_comparison_corr.png', dpi=300)
