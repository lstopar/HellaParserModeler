import numpy as np
import csv
from threading import Thread
import matplotlib.pyplot as plt
from math import exp
from sklearn.linear_model.ridge import Ridge

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def readData(file, target_scrap):
    fin = open(file, "r", encoding = "utf-8")
    
    shift_idx = 2
    scrap_offset = 4
    readings_offset = 50
    
    reader = csv.reader(fin, delimiter=',', quotechar='"')
    rows = [row for row in reader]
    
    headers = rows[0]
    scrap_headers = headers[scrap_offset:readings_offset]
    feature_headers = headers[readings_offset:]
    rows = rows[1:]
    
    shifts = []
    scraps = []
    curr_shift_id = -1
    for row in rows:
        shift_id = row[shift_idx]
        if shift_id != curr_shift_id:
            curr_shift_id = shift_id
            shifts.append([])
            scraps.append([float(val) for val in row[scrap_offset:readings_offset]])
        shift = shifts[-1]
        shift.append([float(val) for val in row[readings_offset:]])
        
    fin.close()
    
    new_shifts = []
    new_scraps = []
    for i, row in enumerate(scraps):
        total_parts = row[0]
        scrap_parts = row[1]
        
        if total_parts < 200 or scrap_parts == 0 or scrap_parts == total_parts:
            print('Omitting shift ' + str(i))
            continue
        
        if total_parts == 490 or total_parts == 580:    # problematic shifts
            print('Omitting shift ' + str(i))
            continue
        
        new_shifts.append(shifts[i])
        new_scraps.append(scraps[i])
    
    return new_shifts, new_scraps, feature_headers, scrap_headers

def preprocess(shifts, ftr_headers, scrap_headers):
    print('Preprocessing ...')

    ftr_headers.append('intersect')
    for shift in shifts:
        for ftrvec in shift:
            ftrvec.append(1)
    
    for shift_n in range(len(shifts)):
        shifts[shift_n] = np.matrix(shifts[shift_n])
        
    return shifts, ftr_headers, scrap_headers

def extract_scraps(scraps, scrap_headers, target_group):
    resp_idxs = [scrap_headers.index(header) for header in target_group]
    resp_arrs = [np.array([row[resp_idx] for row in scraps]) for resp_idx in resp_idxs]
    
    resp = resp_arrs[0]
    for i in range(1, len(resp_arrs)):
        resp = np.add(resp, resp_arrs[i])
        
    return np.array(resp)

def construct_ftr_mat(shifts, scraps, ftr_headers, scrap_headers, response_headers):
    print('Constructing feature matrix ...')
    
    n_percentiles = 5
    
    ftr_mat = []
    ftr_names = []
    
    for shift_n, shift in enumerate(shifts):
        print('Constructing shift: ' + str(shift_n))
        
        shift_arr = np.array(shift)
        means = np.mean(shift_arr, axis=0)
        medians = np.median(shift_arr, axis=0)
        mns = np.amin(shift_arr, axis=0)
        mxs = np.amax(shift_arr, axis=0)
        
        perc = np.linspace(1.0 / (n_percentiles+1), 1 - 1.0 / (n_percentiles+1), n_percentiles)
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
                
#             time_sum = 0;
#             n_peaks = 0
#             mx_time = -1
#             mn_time = -1
#             for inst_n in range(1, shift.shape[0]-1):
#                 if shift[inst_n-1, ftr_n] < shift[inst_n, ftr_n] and shift[inst_n, ftr_n] > shift[inst_n+1, ftr_n]:
#                     # in the maximum
#                     mx_time = inst_n
#                     
#                     if mn_time != -1:
#                         time_sum += mx_time - mn_time
#                         n_peaks += 1
#                 if shift[inst_n-1, ftr_n] > shift[inst_n, ftr_n] and shift[inst_n, ftr_n] < shift[inst_n+1, ftr_n]:
#                     # in the minimum
#                     mn_time = inst_n
#                     
#                     if mx_time != -1:
#                         time_sum += mn_time - mn_time
#                         n_peaks += 1
                    
            mean_diff = diff_sum / (len(shift) - 1)
            #mean_peak_time = float(time_sum) / n_peaks if n_peaks != 0 else shift.shape[0]
            
            row = np.concatenate((row, [mean_diff]))
        
        ftr_mat.append(row)
        
    ftr_names += [name + ' (mean)' for name in ftr_headers]
    ftr_names += [name + ' (median)' for name in ftr_headers]
    ftr_names += [name + ' (min)' for name in ftr_headers]
    ftr_names += [name + ' (max)' for name in ftr_headers]
    
    for p in perc:
        ftr_names += [name + ' (' + str(p) + '-th percentile)' for name in ftr_headers]
    
    for name in ftr_headers:
        ftr_names.append(name + ' (mean diff)')
        #ftr_names.append(name + ' (t between extremes)')
    
    return np.array(ftr_mat), ftr_names

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

def lss(Y, counts, beta, _lambda):
    loss = 0
    grad = 0
    H = np.zeros((beta.shape[0], beta.shape[0]))
    
    for i, S in enumerate(Y):
        n_i = counts[i]
        
        p = logfun(S, beta)
        pder = np.multiply(p, 1 - p)
        
        D = np.zeros((S.shape[0], S.shape[0]))
        np.fill_diagonal(D, np.multiply(pder, 1 - 2*p))
        
        sum1 = np.sum(p)
        sum2 = S.T * pder
        sum3 = S.T * D * S
        
        diff = n_i - sum1
        
        loss += abs(diff)
        grad -= sgn(diff)*sum2
        H -= sgn(diff)*sum3

    return loss + _lambda*np.dot(beta.T, beta)[0,0], grad + 2*_lambda*beta, H + 2*_lambda*np.eye(beta.shape[0])

def sqloss(Y, counts, beta, _lambda):
    loss = 0
    grad = 0
    H = np.zeros((beta.shape[0], beta.shape[0]))
    
    n = 0
    
    for i, S in enumerate(Y):
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

def minimize_newton(X, counts, lossfun, beta0=None, sing_eps=1e-10, _lambda=1, alpha=.1):
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
        loss, grad, H = lossfun(X, counts, beta, _lambda)
        #eigvals, _ = np.linalg.eig(H)
        
        curr_alpha = 1e-5
        prev_loss_grad = float('inf')
        while True:
            loss_grad, _, _ = lossfun(X, counts, beta - curr_alpha*grad, _lambda)
            
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
        
        loss_newton, _, _ = lossfun(X, counts, beta + dbeta_newton, _lambda)
        
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
             
    loss, _, _ = lossfun(X, counts, beta, _lambda)
    print('Final loss:', loss, ', beta =', str(beta.T))
        
    return beta

def evaluate(S, count, beta):
    print('Evaluating ...')
    
    n = S.shape[0]
    
    pred_prob = 0
    for i in range(n):
        x_i = S[i,:]
        p = logfun(x_i, beta)
        pred_prob += p[0,0]
    
    return pred_prob, 0, 0
    
betas = []    
threads = []
test_Y = []
test_counts = []

def fit_single(X, nn, beta0, sing_eps, _lambda, alpha, i):
    global betas
    
    print('Fitting task:', i)
    betas[i] = minimize_newton(X, nn, sqloss, beta0, sing_eps=sing_eps, _lambda=_lambda, alpha=alpha)
    print('Finished task:', i)
    
def model_and_eval(X, scraps, multi_thread=False):
    global betas
    global threads
    global test_Y
    global test_counts
    
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
    
    betas = [0]*size
    
    for i in range(size):
        X_train = X[:i] + X[i+1:]
        y_train = np.concatenate((scraps[:i], scraps[i+1:]))
        x_test = X[i]
        y_test = scraps[i]
        
        print('Minimizing squared value ...')
        #beta0 = np.matrix([[-9.60239274e-02, -1.13687161e-02, 1.85787899e-02, -1.18373274e-02, 9.05388970e-02, 6.82847851e-02, 1.92573101e-02, 3.25430459e-02, 8.93466083e-03, 2.28929388e-02, 2.33928005e-02, 4.30076223e-02, 7.00975737e-03, 1.57737056e-01, 5.28103155e-02, 1.52398772e-02, -7.66287887e+00]]).T
        #beta0 = np.matrix(np.random.randn(beta0.shape[0])).T
        beta0 = None
        threads.append(Thread(target=fit_single, args=(X_train, y_train, beta0, sing_eps, _lambda, alpha, i)))
        test_Y.append(x_test)
        test_counts.append(y_test)
        
    for thread in threads:
        thread.start()
        if not multi_thread:
            thread.join()
    if multi_thread:
        for thread in threads:
            thread.join()
        
    for i in range(size):
        pred_prob, pred, thres = evaluate(test_Y[i], test_counts[i], betas[i])
        print('prediction:', pred, '\nprobabilistic prediciton:', pred_prob, '\ncount:', y_test, '\nthreshold:', thres)
        
        prob_preds.append(pred_prob)
        preds.append(pred)
        counts.append(test_counts[i])
        thresholds.append(thres)
        
    print('Prob predicitons:\scraps', str(prob_preds))
    print('Real predicitons:\scraps', str(preds))
    print('Thresholds:\scraps', str(thresholds))
    print('Real counts:\scraps', str(counts))
    print('Sing eps: ', sing_eps)
    print('Lambda: ', _lambda)
    
    return counts, prob_preds

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

group1 = ['141 zagon (%)', '14 pike, pikasto (razno) (%)', '15 meglica (mat, siva površina) (%)']
ignore1 = []#[51, 89]

group2 = ['144 ponovni zagon (po čiščenju orodja) (%)', '23 nitke (%)', '1 nezalito, nedolito (%)']
ignore2 = []#[112]

group3 = ['5 opraskano (robot, trak) (%)', '7 žarki, plastika (%)', '6 onesnaženo s tujki (%)']
ignore3 = []#[15]

group4 = ['4 počeno, zvito, deformirano (%)', '428 črna(e) pika(e) (%)']
ignore4 = []#[51]

group_all = group1 + group2 + group3 + group4 + ['172 U - zanka (pentlja, hakeljček) (%)', '9 enojni žarek (%)']
ignore_all = ignore1 + ignore2 + ignore3 + ignore4

target_group = group_all
target_ignore = ignore_all

fname = '/mnt/raidM2T/data/Hella/scrap-data/all/Hella molding - 2016-05-11.csv'
shifts, scraps, ftr_headers, scrap_headers = readData(fname, 0)
shifts, ftr_headers, scrap_headers = preprocess(shifts, ftr_headers, scrap_headers)

ftr_mat, ftr_names = construct_ftr_mat(shifts, scraps, ftr_headers, scrap_headers, target_group)
response = extract_scraps(scraps, scrap_headers, target_group)
pred, real, wgts = eval_aggr_shifts(ftr_mat, response, target_ignore)

print('Weights:')
wgt_nm_pr_v = [(abs(wgts[i]), wgts[i], ftr_names[i]) for i in range(len(wgts))]
wgt_nm_pr_v.sort()
for _, wgt, ftr_name in wgt_nm_pr_v:
    print_str = '{0:20}: {1}'.format(ftr_name, wgt);
    print(print_str)
    


# for i in range(len(target_group)):
#     target_group[i] = target_group[i].replace(' (%)', '')
# 
# response = extract_scraps(scraps, scrap_headers, target_group)
# real, pred = model_and_eval(shifts, response, multi_thread=True)
 
mae = float(sum([abs(pred[i] - real[i]) for i in range(len(pred))])) / len(pred)
print('MAE: ' + str(mae))
print('relative MAE: ' + str(mae / (sum(real) / len(real))))


plt.plot(real, 'g')
plt.plot(pred, 'r')
plt.xlabel('Shift [#]')
plt.xlabel('Scrap [%]')
plt.show()
