import numpy as np
import matplotlib.pyplot as plt
from math import exp
from sklearn.linear_model.ridge import Ridge
import pickle
import os
from sklearn.linear_model.coordinate_descent import Lasso
from hella_model.io import group_all, ignore_all, readData, preprocess,\
    extract_scraps

def construct_ftr_mat(shifts, scraps, ftr_headers, scrap_headers, response_headers, target_ftrs):
    print('Constructing feature matrix ...')
        
    ftr_mat = []
    ftr_names = []
    
    for shift_n, shift in enumerate(shifts):
        print('Constructing shift: ' + str(shift_n))
        
        shift_arr = np.array(shift)
        means = np.mean(shift_arr, axis=0)
        medians = np.median(shift_arr, axis=0)
        variances = np.var(shift_arr, axis=0)
        mns = np.amin(shift_arr, axis=0)
        mxs = np.amax(shift_arr, axis=0)
        
        perc = np.linspace(100.0 / (n_percentiles+1), 100 - 100.0 / (n_percentiles+1), n_percentiles)
        perc_arr = np.array([])
        for p in perc:
            perc_arr = np.concatenate((perc_arr, np.percentile(shift_arr, p, axis=0)))
        
        row = np.concatenate((means, medians, mns, mxs, perc_arr))
        
        n_ftrs = shift.shape[1]
        for ftr_n in range(n_ftrs):
            # calculate the average difference
            diff_sum = 0
            for inst_n in range(1, shift.shape[0]):
                diff = abs(shift[inst_n, ftr_n] - shift[inst_n-1, ftr_n])
                diff_sum += diff
                    
            mean_diff = diff_sum / (len(shift) - 1)
            #mean_peak_time = float(time_sum) / n_peaks if n_peaks != 0 else shift.shape[0]
            
            row = np.concatenate((row, [mean_diff]))
        
        ftr_mat.append(row)
        
    ftr_names += [name + ' (mean)' for name in ftr_headers]
    ftr_names += [name + ' (median)' for name in ftr_headers]
    #ftr_names += [name + ' (variance)' for name in ftr_headers]
    ftr_names += [name + ' (min)' for name in ftr_headers]
    ftr_names += [name + ' (max)' for name in ftr_headers]
    
    for p in perc:
        ftr_names += [name + ' (' + str(p) + '-th percentile)' for name in ftr_headers]
    
    for name in ftr_headers:
        ftr_names.append(name + ' (mean diff)')
        #ftr_names.append(name + ' (t between extremes)')
    
    # filter the features
    if target_ftrs is not None:
        print('Filtering features ...')
        ftr_idxs = [ftr_names.index(ftr_name) for ftr_name in target_ftrs]
    
        new_ftr_names = [ftr_names[idx] for idx in ftr_idxs]
        new_ftr_mat = []
        for inst_n in range(len(ftr_mat)):
            row = ftr_mat[inst_n]
            new_row = [row[idx] for idx in ftr_idxs]
            new_ftr_mat.append(new_row)
            
        return np.array(new_ftr_mat), new_ftr_names
    else:
        return np.array(ftr_mat), ftr_names

def eval_aggr_shifts(X, y, ignore_rows):
    eps = 1e-6
    pred = []
    real = []
    
    for inst_n in ignore_rows:
        X = np.concatenate((X[:inst_n], X[inst_n+1:]))
        y = np.concatenate((y[:inst_n], y[inst_n+1:]))
    
    n = X.shape[0]
    for inst_n in range(n):
        x_i = X[inst_n]
        y_i = y[inst_n]
        
        X_train = np.concatenate((X[:inst_n], X[inst_n+1:]))
        y_train = np.concatenate((y[:inst_n], y[inst_n+1:]))
        
        y_train = np.array([max(eps, min(1 - eps, val)) for val in y_train])
        y_train = np.log(y_train / (1 - y_train))
        
        model = Ridge(alpha=.2, fit_intercept=True, normalize=True)
        #model = Lasso(alpha=.001, fit_intercept=True, normalize=True)
        model.fit(X_train, y_train)
        
        y_hat = model.predict(x_i.reshape(1, -1))[0]
        
        y_i1 = max(eps, min(1 - eps, y_i))
        y_i1 = np.log(y_i1 / (1 - y_i1))
        print('inst: ' + str(inst_n) + ', prediction: ' + str(y_hat) + ', err: ' + str(y_hat - y_i1))
        
        pred.append(1 / (1 + exp(-y_hat)))
        real.append(y_i)
        
    model = Ridge(alpha=.2, fit_intercept=True, normalize=True)
    model.fit(X, y)
        
    return pred, real, model.coef_

load_save_ftrs = True
ftrs_fnm = 'hella-ftrs.p'

target_ftrs = None
loaded = False
if load_save_ftrs and os.path.isfile(ftrs_fnm):
    print('Loading feature names ...')
    pin = open(ftrs_fnm, 'rb')
    target_ftrs = pickle.load(pin)
    loaded = True
    pin.close()

target_group = group_all
target_ignore = ignore_all

fname = '/mnt/raidM2T/data/Hella/scrap-data/all/Hella molding - 2016-05-11.csv'
shifts, scraps, ftr_headers, scrap_headers = readData(fname, 0)
shifts, ftr_headers, scrap_headers = preprocess(shifts, ftr_headers, scrap_headers)

ftr_mat, ftr_names = construct_ftr_mat(shifts, scraps, ftr_headers, scrap_headers, target_group, target_ftrs)
response = extract_scraps(scraps, scrap_headers, target_group)
pred, real, wgts = eval_aggr_shifts(ftr_mat, response, target_ignore)

print('Weights:')
wgt_nm_pr_v = [(abs(wgts[i]), wgts[i], ftr_names[i]) for i in range(len(wgts))]
wgt_nm_pr_v.sort()

target_ftrs = []

for abs_wgt, wgt, ftr_name in wgt_nm_pr_v:
    print_str = '{0:20}: {1}'.format(ftr_name, wgt);
    print(print_str)
    
    if not loaded and abs_wgt > 1e-5:
        target_ftrs.append(ftr_name)
      
if not loaded and load_save_ftrs:  
    pout = open(ftrs_fnm, 'wb')
    pickle.dump(target_ftrs, pout)
    pout.flush()
    pout.close()
    
 
mae = float(sum([abs(pred[i] - real[i]) for i in range(len(pred))])) / len(pred)
print('MAE: ' + str(mae))
print('relative MAE: ' + str(mae / (sum(real) / len(real))))


plt_real = plt.plot(real, 'g')
plt_pred = plt.plot(pred, 'r')
plt.xlabel('Shift [#]')
plt.ylabel('Scrap [%]')
plt.legend(['Real scrap', 'Predicted scrap'])
plt.show()

