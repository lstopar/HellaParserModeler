import numpy as np
## import scipy
from scipy import optimize
## from scipy.optimize import fmin_tnc

## np.test()
## scipy.test()
## optimize.fmin_tnc()


def loss(x0, X, n):
    summ = 0
    i = 0
    for shift in X:
        n1 = n[i]
        i += 1
        vec = shift.dot(x0)
        vec = np.log(1 + np.exp(vec))
        vec = np.sum(vec)
        summ += abs(n1 - vec)
    return summ

def evaluiraj(x0, X, n):
    summ = 0
    i = 0
    ## je samo en shift
    for shift in X:
        n1 = n[i]
        i += 1
        # sum1 = 0
        # for event in shift:
        #     event = [np.log(1+ np.exp(event[i] * x0[i])) for i in range(len(event))]
        #     sum1 += sum(event)
        # summ += (n - sum1) * (n - sum1)
        # print(shift.shape)
        vec = shift.dot(x0)
        vec = np.log(1 + np.exp(vec))
        vec = np.sum(vec)
    return vec

def gradient(x0, X, n):
    summ = np.matrix([0.]* X[0].shape[1])
    i = 0
    for shift in X:
        n1 = n[i]
        i += 1
        # sum1 = 0
        # sum2 = 0
        # for event in shift:
        #     sum2 += sum(event)
        #     event = [np.log(1+ np.exp(event[i] * x0[i])) for i in range(len(event))]
        #     sum1 += sum(event)
        # summ += (n - sum1) * 2 * (- sum2)
        vec = shift.dot(x0)
        vec = np.log(1 + np.exp(vec))
        vec = n1 - np.sum(vec)
        vec1 = np.sum(shift, 0)
        vec1 = np.multiply(vec1, x0)
        summ += np.multiply(np.sign(vec1), vec1)
    summ = np.array(summ)[0]
    return summ

def mySplit(string, seperator = ","):
    if string == "": return []
    res = []
    i = 0
    j = 0
    quoted = False
    for char in string:
        if char == '"':
            quoted = not quoted
        elif char == seperator and not quoted:
            app = string[i:j]
##            if app and app[-1] == '"': app = app[:-1]
##            if app and app[0] == '"': app = app[1:]
            res.append(app)
            i = j + 1 ## don't want to include the seperator
        j += 1
    res.append(string[i:])
    return res

def substractOne(date):
    ## dela samo za trenutne mesece. todo
    days = {1:31, 10:31, 11:30, 12:31, 2:29}
    if date[0] > 1:
        date[0] -= 1
    elif date[1] > 1:
        date[1] -= 1
        date[0] = days[date[1]]
    else:
        date[2] -= 1
        date[0] = 31
        date[1] = 12
    return date

def readData(file = "./result_61282649.csv"):
    f = open(file, "r", encoding = "utf-8")
    i = 0
    X = []
    prejsnjaIzmena = -1
    prejsnjiDatum = -1
    for line in f.readlines():
        i += 1
        line = mySplit(line[:-1], ",")
        if i == 1:
            header = line[1:-2]
        else:
            ## todo. da si lahko izberem katere v vrstici.
            ## da se ne sesuje, ce ni int, ampak proba sparsat
            ## (ker je ponavadi to pri št. delavcev)

            ## vn vržemo komentar/st. delavcev -> zadnja 2
            ## in cloveski timestamp -> prvi
            timestamp = line[0].split(" ")
            date = [int(k) for k in timestamp[0].split("-")]
            time = timestamp[1].split(":")
            hour = int(time[0])
            if hour <= 5:
                izmena = 0
            elif hour > 5 and hour <= 13:
                izmena = 1
            elif hour > 13 and hour <= 21:
                izmena = 2
            elif hour > 21 and hour <= 23:
                izmena = 3
            else:
                print("Nedefinirana ura.")
                raise
            if (prejsnjiDatum != date
                and (prejsnjaIzmena != 3 or
                     izmena != 0 or
                     prejsnjiDatum != substractOne(date))):
                X.append([])
                prejsnjiDatum = date
                prejsnjaIzmena = izmena
            elif izmena != prejsnjaIzmena:
                X.append([])
                prejsnjiDatum = date
                prejsnjaIzmena = izmena
            line1 = [float(k) if k else 0 for k in line[1:-2] ]
            X[-1].append(line1)

    f.close()
    return X,header
	
	
def modeliraj(X, header, izbranaPolja = ['"7 žarki, plastika"', "Max Po(HitBriz)[mm/s]", "Prirobn(Z15)[°C]"]):
	## prvi senzor v izbranaPolja se uporabi samo za testiranje in ne
	## za model (ker ga napovedujemo)
	## izbranaPolja = ['"7 žarki, plastika"', "Max Po(HitBriz)[mm/s]", "Prirobn(Z15)[°C]"]
	izbranaPolja1 = [-1] * len(izbranaPolja)
	for i in range(len(izbranaPolja)):
		izbranaPolja1[i] = header.index(izbranaPolja[i])
	## izbranaPolja1.append(0) ## timestamp
	k = izbranaPolja1[0]
	izbranaPolja1 = izbranaPolja1[1:]

	n = [izmena[0][k] for izmena in X]
	print(n)
	print(k, header[k])

	## dodamo konstanto, da imamo afine funkcije
	izbranaPolja1.append(len(X[0][0]))
	for i in range(len(X)):
		for j in range(len(X[i])):
			X[i][j].append(1)

	## X = [[[line[j] for j in izbranaPolja1] for line in izmena] for izmena in X]
	## Y = [[[ np.sign(line[j]) * np.log(1 + abs(line[j])) for j in izbranaPolja1] for line in izmena] for izmena in X]
	Y = [[[ np.log(1 + abs(line[j])) for j in izbranaPolja1] for line in izmena] for izmena in X]
	## Y = [[[ line[j] for j in izbranaPolja1] for line in izmena] for izmena in X]




	for izmena in range(len(Y)):
		Y[izmena] = np.matrix(Y[izmena])

	## test
	size = len(Y)
	size1 = Y[0].shape[1]

	eps = np.finfo(float).eps

	print("Start")
	metode = ['Nelder-Mead','Powell','CG','BFGS','Newton-CG',
			  'L-BFGS-B','TNC','COBYLA','SLSQP'] #,'dogleg','trust-ncg'] rabita hessiana
	metode = ['Nelder-Mead','Powell','COBYLA'] # najboljše metode
	# for i in range(size):
	napake = [0] * len(metode)
	#for meth in range(len(metode)):
	meth = 2
	for i in range(size):
		if i < size - 1:
			YY = Y[:i] + Y[i+1:]
			nn = n[:i] + n[i+1:]
		else:
			XX = Y[:i]
			nn = n[:i]
		rest = Y[i]
		nrest = n[i]

		# res = optimize.fmin_tnc(lambda x: loss(x, XX, k),
		#                               [1]*len(X[0][0]),
		#                               lambda x: gradient(x, XX, k))
		res = optimize.minimize(lambda x: loss(x, YY, nn),
								[0]*size1,
								jac = lambda x: gradient(x, YY, n),
								method = metode[meth] )
		# print(res)
		# print("Napaka: ", loss(res.x, YY, n))
		# print("Napaka na testni: ", loss(res.x, [rest], [nrest]))
		# print("Evaluacija na testni: ", evaluiraj(res.x, [rest], [nrest]))
		# print("Scrap na testni: ", n[i])
		# print()
		napake[meth] += abs(n[i] - evaluiraj(res.x, [rest], [nrest]))
	print("Model: ",izbranaPolja[0])
	print("Metoda: ",metode[meth])
	print("Napaka: ",napake[meth])
	print("Izmerjen scrap: ",sum(n))
	print()
	return




## X = podatki v matriki
## y = -1 oz. 1

## X = np.matrix(array like)
X, header = readData()


modeli = {('Under-moulded', 'Sink mark', 'Cold injection'):
              ['Temperature of the hot rummer system',
               'Temperature of the mould',
               'Injection speed',
               'Injection pressure',
               'Holding pressure',
               'Time of holding pressure',
               'Material temperature'
               'Temperature of the cylinder',
               'Switch point'],
          ('Flash'):
              ['Temperature of the mould',
               'Injection speed',
               'Injection pressure',
               'Mould damage',
               'Clamping force',
               'Material temprature'],
          ('Flow lines', 'Flow hook'):
              ['Mould damage',
               'Temperature of the mould',
               'Injection speed',
               'Cylinder damage',
               'Mould venting',
               'Foreign particles in the mould',
               'Material temperature',
               'Hot runner system damage'],
          ('Cracked'):
              ['Mould damage',
               'Robot, transport line damage'],
          ('Scratched surface'):
              ['Robot, transport line damage'],
          ('Foreign particles', 'Dots'):
              ['Dust on the floor shop'],
          ('Welding line'):
              ['Injection speed',
               'Temperature of the mould',
               'Mould damage',
               'Material temperature'],
          ('Bubbles'):
              ['Injection speed',
               'Holding pressure',
               'Time of holding pressure'],
          ('Opaque surface'):
              ['Injection speed',
               'Mould venting',
               'Dust in the mould'],
          ('Hairs'):
              ['Cooling time',
               'Material temperature',
               'Decompression time'],
          ('Drag marks'):
              ['Mould damage'],
          ('Black dots'):
              ['Cylinder damage',
               'Cylinder overheated',
               'Hot runner system damage']
          }

		  
for modell,izbranaPolja in modeli.items():
	for model in modell:
		izbranaPolja2 = [model] + izbranaPolja
		modeliraj(X,header,izbranaPolja2)
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  
		  



