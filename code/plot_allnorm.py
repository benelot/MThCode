import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':

    pre = 'allpos_rnn'
    patient_id = 'ID07'

    ids = []
    h_offset = 31
    for h_ in range(9): # 9
        for m in range(12):
            h = h_ + h_offset
            m = 5 * m

            zero = ''
            if m < 10:
                zero = '0'

            t_string = str(h) + 'h' + zero + str(m) + 'm'
            ids.append(pre + '_' + patient_id + '_' + t_string)

    mean_abs, mse, mae, corr = ufig.mean_weights(ids=ids, save_name=pre, hidden=False, diagonal=True)
    mean_abs_true, _, _, _ = ufig.mean_weights(ids=ids, save_name=pre, hidden=True, diagonal=True)

    test = 1

