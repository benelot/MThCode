
# import ieeg_utilities as ieeg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler


def determine_sample_size(patient_id: list=None, time_begin: list=None, max_sample_size: float=None, dt: float=None,
                          save_name='default', load_name=None):
    """

    :param patient_id: List with string or multiple strings.
    :param time_begin: List with list with [hour, minute] or multiple lists.
    :param max_sample_size: In seconds.
    :param dt: In seconds.
    :param save_name: String.
    :param load_name: If not None, only plot is applied.
    :return:
    """
    if load_name is None:
        corr_dts = []
        t_size = np.arange(dt, max_sample_size, dt).tolist()
        for i, id_ in enumerate(patient_id):
            print('Computes job ' + str(i) + '/' + str(len(patient_id) - 1))
            # Load and prepare data
            data_mat = loadmat('../data/' + id_ + '_' + str(time_begin[i][0]) + 'h.mat')
            info_mat = loadmat('../data/' + id_ + '_info.mat')
            fs = float(info_mat['fs'])
            sample_begin = int(time_begin[i][1] * 60 * fs)
            sample_end = sample_begin + int(max_sample_size * fs)
            data_raw = data_mat['EEG'][:, sample_begin:sample_end].transpose()

            # Get correlation matrices
            corr = np.zeros((len(t_size) + 1, data_raw.shape[1], data_raw.shape[1]))
            corr_dt = np.zeros((len(t_size), data_raw.shape[1], data_raw.shape[1]))
            sc = StandardScaler()
            for j, t in enumerate(t_size):
                data_norm = sc.fit_transform(data_raw[:int(t * fs), :])
                corr[j + 1, :, :] = np.corrcoef(data_norm.T)
                corr_dt[j, :, :] = corr[j + 1, :, :] - corr[j, :, :]

            corr_dts.append(np.sum(np.sum(np.abs(corr_dt), axis=1), axis=1))

        # Make DataFrame
        df = pd.DataFrame()
        for i, id_ in enumerate(patient_id):
            sub_df = pd.DataFrame()
            sub_df['unique_id'] = (np.ones(len(t_size)) * i).astype(int)
            sub_df['Patient ID'] = [id_ for _ in range(len(t_size))]
            sub_df['dt'] = [dt for _ in range(len(t_size))]
            sub_df['corr_dt'] = corr_dts[i]
            sub_df['t_size'] = t_size
            df = df.append(sub_df, ignore_index=True)
        df.to_pickle('../data/sample_size_det_' + save_name + '.pkl')

    else:
        df = pd.read_pickle('../data/sample_size_det_' + load_name + '.pkl')

    # Plot results
    sns.set_style('white')
    plt.figure(figsize=(8, 5))
    sns.lineplot(x='t_size', y='corr_dt', data=df, hue='Patient ID')
    plt.xlabel('Sample size [s]'), plt.ylabel('Sum of abs. weight changes [-]')
    plt.title('Weight change per ' + str(df['dt'][0]) + ' sec.')
    plt.ylim(0, df.quantile(0.97)['corr_dt']), plt.xlim(df['t_size'].min(), df['t_size'].max())
    plt.savefig('../doc/figures/sample_size_det_' + save_name + '.png')
    plt.close()


patient_id = ['ID07', 'ID07', 'ID07', 'ID08', 'ID08', 'ID08', 'ID11', 'ID11', 'ID11', 'ID11', 'ID11', 'ID11']
time_begin = [[32, 0], [35, 0], [38, 0], [58, 0], [60, 0], [64, 30], [60, 0], [62, 0], [65, 0], [129, 40],
              [132, 30], [136, 30]]

determine_sample_size(patient_id, time_begin, max_sample_size=60, dt=5)


# ieeg.plot_corr_connectivity(r2, r2_dt, h2, h2_dt, save_name=name)
# ieeg.make_corr_connectivity(patient_id[i], time_begin[i], duration, t_shift=0.05, plot_name=patient_id[i] + '_0')

