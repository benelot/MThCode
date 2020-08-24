import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import os

if __name__ == '__main__':

    pre = 'allpos'
    patient_id = 'ID11b'

    ids = []
    h_offset = 129
    for h_ in range(9):
        for m in range(30):
            h = h_ + h_offset
            m = 2 * m

            zero = ''
            if m < 10:
                zero = '0'

            t_string = str(h) + 'h' + zero + str(m) + 'm'
            ids.append(pre + '_' + patient_id + '_' + t_string)

    ufig.mean_weights(ids=ids, save_name=pre, hidden=False, diagonal=True)


