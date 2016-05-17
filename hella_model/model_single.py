import numpy as np
from threading import Thread
from hella_model.io import readData, preprocess, group_all, ignore_all,\
    extract_scraps, group_single, ignore_single, n_percentiles
from os.path import os
import pickle
import re
from multiprocessing import Process, Array
import matplotlib.pyplot as plt

def sgn(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0

def logfun(X, beta):
    dot = np.minimum(-(X * beta), 700)
    return 1 / (1 + np.exp(dot))

# def lss(Y, counts, beta, _lambda):
#     loss = 0
#     grad = 0
#     H = np.zeros((beta.shape[0], beta.shape[0]))
#     
#     for i, S in enumerate(Y):
#         n_i = counts[i]
#         
#         p = logfun(S, beta)
#         pder = np.multiply(p, 1 - p)
#         
#         D = np.zeros((S.shape[0], S.shape[0]))
#         np.fill_diagonal(D, np.multiply(pder, 1 - 2*p))
#         
#         sum1 = np.sum(p)
#         sum2 = S.T * pder
#         sum3 = S.T * D * S
#         
#         diff = n_i - sum1
#         
#         loss += abs(diff)
#         grad -= sgn(diff)*sum2
#         H -= sgn(diff)*sum3
# 
#     return loss + _lambda*np.dot(beta.T, beta)[0,0], grad + 2*_lambda*beta, H + 2*_lambda*np.eye(beta.shape[0])

def sqloss(X, counts, beta, _lambda, calc_grad):
    if calc_grad:
        loss = 0
        grad = 0
        H = np.zeros((beta.shape[0], beta.shape[0]))
        
        n = 0
        
        for i, S in enumerate(X):
            n_i = counts[i]
            
            n += S.shape[0]
            
            p = logfun(S, beta)
            pder = np.multiply(p, 1 - p)
            
            D = np.zeros((S.shape[0], S.shape[0]))
            np.fill_diagonal(D, np.multiply(pder, 1 - 2*p))
            
            sum1 = np.sum(p)
            sum2 = S.T * pder
            sum3 = S.T * D * S
            
            diff = n_i - sum1
            
            loss += diff*diff
            grad += sum2*diff
            H += sum2*sum2.T - diff*sum3
            
        reg_loss = np.dot(beta[0:-1].T, beta[0:-1])[0,0] / 2
        reg_grad = _lambda*beta
        reg_hesse = _lambda*np.eye(beta.shape[0])
        
        # do not punish the itercept
        reg_grad[-1,0] = 0
        reg_hesse[-1,-1] = 0
        
        return (loss / n + reg_loss) / 2, reg_grad - grad / n, H / n + reg_hesse
    else:
        loss = 0
        n = 0
        
        for i, S in enumerate(X):
            n_i = counts[i]
            n += S.shape[0]
            
            p = logfun(S, beta)            
            diff = n_i - np.sum(p)
            
            loss += diff*diff
            
        reg_loss = np.dot(beta[0:-1].T, beta[0:-1])[0,0] / 2
        
        return (loss / n + reg_loss) / 2

def svd_solve(A, b, eps=0):
    U, sigma, V = np.linalg.svd(A, full_matrices=False)
        
    k = 0
    while k + 1 < len(sigma) and sigma[k+1] > sigma[0]*eps:
        k += 1
    k += 1
    
    Sinv = np.matrix(np.diag(1 / sigma[0:k]))
    U = np.matrix(U[:,0:k])
    V = np.matrix(V[0:k,:])
    
    Ainv = V.T * Sinv * U.T
    return Ainv * b

def minimize(X, counts, lossfun, beta0=None, sing_eps=1e-10, _lambda=1, alpha=.1):
    eps = 1e-7
    
    beta = None
    if beta0 is None:
        beta = np.matrix(np.zeros(X[0].shape[1])).T
        for i in range(X[0].shape[1]):
            beta[i,0] = (np.random.random() - .5)*10
    else:
        beta = beta0
        
    change = float('inf')
    
    while change > eps:
        loss, grad, H = lossfun(X, counts, beta, _lambda, True)
        #eigvals, _ = np.linalg.eig(H)
        
        curr_alpha = 1e-5
        prev_loss_grad = float('inf')
        while True:
            loss_grad = lossfun(X, counts, beta - curr_alpha*grad, _lambda, False)
            
            print('delta loss: ', (loss_grad - loss), ', alpha: ', curr_alpha)
            
            if curr_alpha > 10000:
                break
            if loss_grad > prev_loss_grad:
                curr_alpha /= 2
                loss_grad = prev_loss_grad
                break
            
            curr_alpha *= 2
            prev_loss_grad = loss_grad
            
        dbeta_grad = -curr_alpha*grad
        dbeta_newton = svd_solve(H, -grad, sing_eps)
        
        loss_newton = lossfun(X, counts, beta + dbeta_newton, _lambda, False)
        
        print('loss gradient:', loss_grad, ', loss newton: ', loss_newton)
        
        new_loss = min(loss_grad, loss_newton)
        delta_loss = new_loss - loss
        
        if loss - new_loss < 1e-6:
            print('Loss not decreasing, terminating ...')
            break
        
        if loss_grad < loss_newton:
            print('Using gradient step ...')
            change = np.linalg.norm(dbeta_grad, ord=np.inf)
            beta += dbeta_grad
            print('loss =', new_loss, ', delta loss:', delta_loss, ', norm =', change)
        else:
            print('Using Newton step ...')
            change = np.linalg.norm(dbeta_newton, ord=np.inf)
            beta += dbeta_newton
            print('loss =', new_loss, ', delta loss:', delta_loss,', norm =', change)
             
    print('Final loss:', new_loss, ', beta =', str(beta.T))
        
    return beta
    
def ftr_engineer(X_train, X_test, ftr_headers, target_ftrs):
    print('Feature engineering ...')
    
    window_size = 15
    
    for i in range(len(X_train)):
        X_train[i] = np.array(X_train[i])
    X_test = np.array(X_test)
    
    all_readings = []
    for shift in X_train:
        for row in shift:
            all_readings.append(row)
    
    conf_h = {}
    for ftr_conf in target_ftrs:
        search = re.search('\([\w\W]*\)', ftr_conf)
        
        if search is None:
            print('Could not match feature: ' + str(ftr_conf))
            exit()
        
        conf = search.group(0)
        name = ftr_conf[:len(ftr_conf) - len(conf) - 1]
        
        if not name in ftr_headers:
            print('Feature ' + name + ' missing in headers!!')
            exit()
        
        if not name in conf_h:
            conf_h[name] = []
        
        conf_h[name].append(conf)
        
    new_shifts = []
    for shift in shifts:
        new_shift = []
        for row in shift:
            new_shift.append([])
        new_shifts.append(new_shift)
    
    new_test = [[] for _ in range(len(X_test))]
    new_headers = []
    for i in range(len(X_train)):
        new_shifts.append([])
        for _ in range(len(X_train[i])):
            new_shifts[i].append([])
            
    for ftr_n, ftr_name in enumerate(ftr_headers):
        print('Engineering feature: ' + ftr_name + ' (' + str(ftr_n+1) + ' of ' + str(len(ftr_headers)) + ')')
        
        if not ftr_name in conf_h:
            print('Skipping ...')
            continue
        if ftr_name == 'intercept':
            print('Omitting intercept!')
            continue
        
        new_headers.append(ftr_name)
        for shift_n in range(len(X_train)):
            for inst_n in range(len(X_train[shift_n])):
                new_shifts[shift_n][inst_n].append(X_train[shift_n][inst_n][ftr_n])
                
        for row_n in range(len(X_test)):
            new_test[row_n].append(X_test[row_n][ftr_n])
            
        has_percentiles = False
                
        confs = conf_h[ftr_name]
        for conf in confs:
            print('\t' + conf)
            
            if not has_percentiles and conf.endswith('percentile)'):
                has_percentiles = True
                
                res = re.search('\d+\.\d+', conf)
                if res is None:
                    print('Could not match percentile: ' + str(conf))
                    exit()
                p_str = res.group(0)
                
                new_headers.append(ftr_name + ' (' + str(p_str) + ' percentile)')
                
                p = float(p_str)
                vals = [all_readings[i][ftr_n] for i in range(len(all_readings))]
                
                percentiles = np.linspace(100.0 / (n_percentiles+1), 100 - 100.0 / (n_percentiles+1), n_percentiles)
                perc_arr = []
                for p in percentiles:
                    perc_val = np.percentile(vals, p)
                    perc_arr.append(perc_val)
                    
                for shift_n, shift in enumerate(X_train):
                    for inst_n, row in enumerate(shift):
                        val = row[ftr_n]
                        perc_n = 0
                        ftr_v = [0]*(len(perc_arr)+1)
                        while perc_n < len(perc_arr) and val < perc_arr[perc_n]:
                            perc_n += 1
                        ftr_v[perc_n] = 1
                        for ftr_val in ftr_v:
                            new_shifts[shift_n][inst_n].append(ftr_val)
                    
                # update the training shift
                for row_n, row in enumerate(X_test):
                    val = row[ftr_n]
                    ftr_v = [0]*(len(perc_arr)+1)
                    while perc_n < len(perc_arr) and val < perc_arr[perc_n]:
                        perc_n += 1
                    ftr_v[perc_n] = 1
                    for ftr_val in ftr_v:
                        new_test[row_n].append(ftr_val)
                        
            elif conf.endswith('(mean diff)'):
                new_headers.append(ftr_name + ' (mean diff)')
                for shift_n, shift in enumerate(X_train):
                    for inst_n in range(len(shift)):
                        diffs = [abs(shift[i][ftr_n] - shift[i-1][ftr_n]) for i in range(max(0, inst_n - window_size), inst_n+1)]
                        mean_diff = sum(diffs) / len(diffs) if len(diffs) > 0 else 0
                        new_shifts[shift_n][inst_n].append(mean_diff)
                    
                # update the training shift
                for row_n, row in enumerate(X_test):
                    diffs = [abs(X_test[i][ftr_n] - X_test[i-1][ftr_n]) for i in range(max(0, row_n - window_size), row_n+1)]
                    mean_diff = sum(diffs) / len(diffs) if len(diffs) > 0 else 0
                    new_test[row_n].append(mean_diff)
                    
            elif conf.endswith('(mean)'):
                new_headers.append(ftr_name + ' (mean)')
                for shift_n, shift in enumerate(X_train):
                    for inst_n in range(len(shift)):
                        vals = [shift[i][ftr_n] for i in range(max(0, inst_n - window_size), inst_n+1)]
                        mean = sum(vals) / len(vals) if len(vals) > 0 else 0
                        new_shifts[shift_n][inst_n].append(mean)
                
                # update the training shift
                for row_n, row in enumerate(X_test):
                    vals = [X_test[i][ftr_n] for i in range(max(0, row_n - window_size), row_n+1)]
                    mean = sum(vals) / len(vals) if len(vals) > 0 else 0
                    new_test[row_n].append(mean)
                    
            elif conf.endswith('(median)'):
                new_headers.append(ftr_name + ' (median)')
                for shift_n, shift in enumerate(X_train):
                    for inst_n in range(len(shift)):
                        vals = [shift[i][ftr_n] for i in range(max(0, inst_n - window_size), inst_n+1)]
                        median = np.median(vals) if len(vals) > 0 else 0
                        new_shifts[shift_n][inst_n].append(median)
                
                # update the training shift
                for row_n, row in enumerate(X_test):
                    vals = [X_test[i][ftr_n] for i in range(max(0, row_n - window_size), row_n+1)]
                    median = np.median(vals) if len(vals) > 0 else 0
                    new_test[row_n].append(median)
                
            elif conf.endswith('(max)'):
                new_headers.append(ftr_name + ' (max)')
                for shift_n, shift in enumerate(X_train):
                    for inst_n in range(len(shift)):
                        vals = [shift[i][ftr_n] for i in range(max(0, inst_n - window_size), inst_n+1)]
                        new_shifts[shift_n][inst_n].append(np.max(vals) if len(vals) > 0 else 0)
                    
                # update the training shift
                for row_n, row in enumerate(X_test):
                    vals = [X_test[i][ftr_n] for i in range(max(0, row_n - window_size), row_n+1)]
                    new_test[row_n].append(np.max(vals) if len(vals) > 0 else 0)
                    
            elif conf.endswith('(min)'):
                new_headers.append(ftr_name + ' (min)')
                for shift_n, shift in enumerate(X_train):
                    for inst_n in range(len(shift)):
                        vals = [shift[i][ftr_n] for i in range(max(0, inst_n - window_size), inst_n+1)]
                        new_shifts[shift_n][inst_n].append(np.min(vals) if len(vals) > 0 else 0)
                    
                # update the training shift
                for row_n, row in enumerate(X_test):
                    vals = [X_test[i][ftr_n] for i in range(max(0, row_n - window_size), row_n+1)]
                    new_test[row_n].append(np.min(vals) if len(vals) > 0 else 0)
            else:
                print('Unknown conf: ' + conf)
                exit()
                
    new_headers.append('intercept')
    for i in range(len(new_shifts)):
        shift = new_shifts[i]
        for j in range(len(shift)):
            row = shift[j]
            row.append(1)
        new_shifts[i] = np.matrix(new_shifts[i])
        print('=================================================')
        print(str(new_shifts[i]))
        print('=================================================')
        
    for row_n, row in enumerate(X_test):
        new_test[row_n].append(1)
        
    return new_shifts, np.matrix(new_test), new_headers
    
processes = []
test_counts = []
results = None

def evaluate(S, count, beta):
    print('Evaluating ...')
    
    n = S.shape[0]
    
    pred_prob = 0
    for i in range(n):
        x_i = S[i,:]
        p = logfun(x_i, beta)
        pred_prob += p[0,0]
    
    return pred_prob, 0, 0

def fit_single(X_train, y_train, X_test, n_test, beta0, sing_eps, _lambda, alpha, i, ftr_headers, results):    
    X_train, X_test, ftr_headers = ftr_engineer(X_train, X_test, ftr_headers, target_ftrs)
    
    print('Fitting task:', i)
    #betas[i] = minimize(X_train, y_train, sqloss, beta0, sing_eps=sing_eps, _lambda=_lambda, alpha=alpha)
    beta = minimize(X_train, y_train, sqloss, beta0, sing_eps=sing_eps, _lambda=_lambda, alpha=alpha)
    pred_prob, _, _ = evaluate(X_test, n_test, beta)
    results[i] = pred_prob
    print('Finished task: ' + str(i) + ', real: ' + str(n_test) + ', pred: ' + str(pred_prob) + ', diff = ' + str(abs(pred_prob - n_test)) + ', beta: ' + str(beta))
    
def model_and_eval(X, y, ftr_headers, thread_count=1):
    global processes
    global test_counts
    global results
    
    print('Fitting models ...')
    
    ## test
    size = len(X)


    print("Start")
    
    prob_preds = []
    preds = []
    counts = []
    thresholds = []
    
    sing_eps = 1e-9
    _lambda = .001#1#0#.001        # regularization
    alpha = .1                      # gradient descent learning rate
    
    results = Array('d', size)
    
    for i in range(size):
        X_train = X[:i] + X[i+1:]
        y_train = np.concatenate((y[:i], y[i+1:]))
        x_test = X[i]
        y_test = y[i]
        
        print('Minimizing squared value ...')
        #beta0 = np.matrix([[-9.60239274e-02, -1.13687161e-02, 1.85787899e-02, -1.18373274e-02, 9.05388970e-02, 6.82847851e-02, 1.92573101e-02, 3.25430459e-02, 8.93466083e-03, 2.28929388e-02, 2.33928005e-02, 4.30076223e-02, 7.00975737e-03, 1.57737056e-01, 5.28103155e-02, 1.52398772e-02, -7.66287887e+00]]).T
        #beta0 = np.matrix(np.random.randn(beta0.shape[0])).T
        beta0 = None
        processes.append(Process(target=fit_single, args=(X_train, y_train, x_test, y_test, beta0, sing_eps, _lambda, alpha, i, ftr_headers, results)))
        test_counts.append(y_test)
        

    curr_task = 0
    while curr_task < len(processes):
        started_threads = []
        for i in range(thread_count):
            print('Starting task ' + str(curr_task))
            thread = processes[curr_task]
            
            started_threads.append(thread)
            thread.start()
            
            curr_task += 1
            if curr_task >= len(processes):
                break
        for thread in started_threads:
            thread.join()
        
    X_train, _, ftr_headers = ftr_engineer(X, [], ftr_headers, target_ftrs)
    beta = minimize(X_train, y, sqloss, None, sing_eps=sing_eps, _lambda=_lambda, alpha=alpha)
        
    res = results[:]
    for i in range(size):
        pred_prob = res[i]
        y_test = test_counts[i]
        #pred_prob, pred, thres = evaluate(test_Y[i], test_counts[i], betas[i])
        print('probabilistic prediciton:', pred_prob, ', count:', y_test)
        
        prob_preds.append(pred_prob)
        counts.append(y_test)
        
    print('Prob predicitons:\y', str(prob_preds))
    print('Real predicitons:\y', str(preds))
    print('Thresholds:\y', str(thresholds))
    print('Real counts:\y', str(counts))
    print('Sing eps: ', sing_eps)
    print('Lambda: ', _lambda)
    
    return counts, prob_preds, beta, headers

load_save_ftrs = True
ftrs_fnm = 'hella-ftrs.p'

target_ftrs = None
if load_save_ftrs and os.path.isfile(ftrs_fnm):
    print('Loading feature names ...')
    pin = open(ftrs_fnm, 'rb')
    target_ftrs = pickle.load(pin)
    loaded = True
    pin.close()

target_group = group_single
target_ignore = ignore_single

fname = '/mnt/raidM2T/data/Hella/scrap-data/all/Hella molding - 2016-05-11.csv'
shifts, scraps, ftr_headers, scrap_headers = readData(fname, 0)
shifts, ftr_headers, scrap_headers = preprocess(shifts, ftr_headers, scrap_headers)

print('Using ' + str(len(ftr_headers)) + ' features ...')

for i in range(len(target_group)):
    target_group[i] = target_group[i].replace(' (%)', '')
 
response = extract_scraps(scraps, scrap_headers, target_group)

#real, pred, beta, headers = model_and_eval(shifts, response, ftr_headers, thread_count=10)
real, pred, beta, headers = model_and_eval(shifts, response, ftr_headers, thread_count=1)

wgt_nm_pr_v = [(abs(beta[i,0]), headers[i], beta[i,0]) for i in range(len(headers))]
wgt_nm_pr_v.sort()

print('================================================')
print('Results:')
for _, ftr_name, wgt in wgt_nm_pr_v:
    print(ftr_name + ':\t' + str(wgt))
print('================================================')

plt_real = plt.plot(real, 'g')
plt_pred = plt.plot(pred, 'r')
plt.xlabel('Shift [#]')
plt.ylabel('Scrap [%]')
plt.legend(['Real scrap', 'Predicted scrap'])
plt.show()