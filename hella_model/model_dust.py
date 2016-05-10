import numpy as np
import csv
from threading import Thread
import random

def readData(file, target_scrap):
    fin = open(file, "r", encoding = "utf-8")
    
    shift_idx = 2
    scrap_offset = 4
    readings_offset = 27
    
    reader = csv.reader(fin, delimiter=',', quotechar='"')
    rows = [row for row in reader]
    
    header = rows[0]
    rows = rows[1:]
    
    shifts = []
    scraps = []
    curr_shift_id = -1
    for row in rows:
        shift_id = row[shift_idx]
        if shift_id != curr_shift_id:
            curr_shift_id = shift_id
            shifts.append([])
            scraps.append(float(row[scrap_offset]))
        
        shift = shifts[-1]
        shift.append([float(val) for val in row[readings_offset:]])
        
    fin.close()
    
    return shifts, scraps, header

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
    
    #n_thresholds = 1000
    #thresholds = np.linspace(0, 1, n_thresholds)
    #pred = np.array([0]*n_thresholds)
    
    n = S.shape[0]
    
    pred_prob = 0
    for i in range(n):
        x_i = S[i,:]
        p = logfun(x_i, beta)
        pred_prob += p[0,0]
        
        #for k in range(n_thresholds):
        #    pred[k] += 1 if p > thresholds[k] else 0 
    
    #best_abs = float('inf')
    #best_pred = float('inf')
    #best_threshold = float('inf')
    #for k in range(n_thresholds):
    #    if np.abs(count - pred[k]) < best_abs:
    #        best_abs = np.abs(count - pred[k])
    #        best_pred = pred[k]
    #        best_threshold = thresholds[k]
    
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
    
def preprocess(shifts, header):
    print('Preprocessing ...')

    header.append('intersect')
    for shift in shifts:
        for ftrvec in shift:
            ftrvec.append(1)
    
    for shift_n in range(len(shifts)):
        shifts[shift_n] = np.matrix(shifts[shift_n])
        
    return shifts, header
        
    
    
def model_and_eval(X, scraps, multi_thread=False):
    global betas
    global threads
    global test_Y
    global test_counts
    
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
        if i < size - 1:
            YY = X[:i] + X[i+1:]
            nn = scraps[:i] + scraps[i+1:]
        else:
            YY = X[:i]
            nn = scraps[:i]
        rest = X[i]
        nrest = scraps[i]
        
        print('Minimizing squared value ...')
        #beta0 = np.matrix([[-9.60239274e-02, -1.13687161e-02, 1.85787899e-02, -1.18373274e-02, 9.05388970e-02, 6.82847851e-02, 1.92573101e-02, 3.25430459e-02, 8.93466083e-03, 2.28929388e-02, 2.33928005e-02, 4.30076223e-02, 7.00975737e-03, 1.57737056e-01, 5.28103155e-02, 1.52398772e-02, -7.66287887e+00]]).T
        #beta0 = np.matrix(np.random.randn(beta0.shape[0])).T
        beta0 = None
        threads.append(Thread(target=fit_single, args=(YY, nn, beta0, sing_eps, _lambda, alpha, i)))
        test_Y.append(rest)
        test_counts.append(nrest)
        
    for thread in threads:
        thread.start()
        if not multi_thread:
            thread.join()
    if multi_thread:
        for thread in threads:
            thread.join()
        
    for i in range(size):
        pred_prob, pred, thres = evaluate(test_Y[i], test_counts[i], betas[i])
        print('prediction:', pred, '\nprobabilistic prediciton:', pred_prob, '\ncount:', nrest, '\nthreshold:', thres)
        
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
        
    #print("Model: ",choosenFields[0])
    #print("Metoda: ",metode[meth])
    #print("Napaka: ",napake[meth])
    #print("Izmerjen scraps: ",sum(scraps))
    #print()
# 	return




## X = podatki v matriki
## y = -1 oz. 1

## X = np.matrix(array like)

# modeli = {('Under-moulded', 'Sink mark', 'Cold injection'):
#               ['Temperature of the hot rummer system',
#                'Temperature of the mould',
#                'Injection speed',
#                'Injection pressure',
#                'Holding pressure',
#                'Time of holding pressure',
#                'Material temperature'
#                'Temperature of the cylinder',
#                'Switch point'],
#           ('Flash'):
#               ['Temperature of the mould',
#                'Injection speed',
#                'Injection pressure',
#                'Mould damage',
#                'Clamping force',
#                'Material temprature'],
#           ('Flow lines', 'Flow hook'):
#               ['Mould damage',
#                'Temperature of the mould',
#                'Injection speed',
#                'Cylinder damage',
#                'Mould venting',
#                'Foreign particles in the mould',
#                'Material temperature',
#                'Hot runner system damage'],
#           ('Cracked'):
#               ['Mould damage',
#                'Robot, transport line damage'],
#           ('Scratched surface'):
#               ['Robot, transport line damage'],
#           ('Foreign particles', 'Dots'):
#               ['Dust on the floor shop'],
#           ('Welding line'):
#               ['Injection speed',
#                'Temperature of the mould',
#                'Mould damage',
#                'Material temperature'],
#           ('Bubbles'):
#               ['Injection speed',
#                'Holding pressure',
#                'Time of holding pressure'],
#           ('Opaque surface'):
#               ['Injection speed',
#                'Mould venting',
#                'Dust in the mould'],
#           ('Hairs'):
#               ['Cooling time',
#                'Material temperature',
#                'Decompression time'],
#           ('Drag marks'):
#               ['Mould damage'],
#           ('Black dots'):
#               ['Cylinder damage',
#                'Cylinder overheated',
#                'Hot runner system damage']
#           }


##for modell,izbranaPolja in modeli.items():
##	for model in modell:
##		fields = [model] + izbranaPolja
##		model_and_eval(X,header,fields)
##		  
# fields = ["429 pike"] + ['"Bin11(8um)"','"Bin12(10um)"','"Bin10(6,5um)"','"Bin13(12um)"',
#                  '"Bin0(0,4um)"','"Bin14(14um)"','"Bin5(1,6um)"','"Bin3(1um)"',
#                  '"Bin7(3um)"','"Bin9(5um)"','"Bin4(1,3um)"','"Bin2(0,8um)"',
#                  '"Bin8(4um)"','"Bin15(16um-17um)"','"Bin1(0,5um)"','"Bin6(2,1um)"']

fname = '/mnt/raidM2T/data/Hella/scrap-data/all/out.csv'
shifts, scraps, header = readData(fname, 0)
shifts, header = preprocess(shifts, header)

sh = []
for i in range(len(shifts)):
    sh.append((shifts[i], scraps[i]))
random.shuffle(sh)

shifts = [shift for shift, scrap in sh]
scraps = [scrap for shift, scrap in sh]

model_and_eval(shifts, scraps, multi_thread=False)



