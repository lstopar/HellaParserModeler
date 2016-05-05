def read(file = "./result_61282649.csv"):
    X = []
    with open(file, "r") as f:
        i = 0
        for line in f.readlines():
            line = mySplit(line[:-1],",")
            if i == 0:
                i = 1
                header = line
                for k in range(len(header)):
                    if header[k][:3] == "428":
                        start = k
                        break
            else:
                line = [int(k) if k else 0 for k in line[start:-2]]
                X.append(line)
    header = header[start:-2]
    return X,header
            
def count(X, header):
    counts = []
    for i in range(len(header)):
        counts.append((sum([X[k][i] for k in range(len(X))]),header[i]))
    counts.sort()
    ## najvec je dobrih kosov
    return counts[::-1]

def showStats(counts):
    dobri = counts[0][0]
    ostali = sum([counts[i][0] for i in range(1,len(counts))])
    print("Dobri: {0}, slabi: {1}, kar je {2} procentov slabih.".format(
        dobri,
        ostali,
        ostali / (ostali + dobri)
        ))
##    print("Najpogostejši: ")
##    for i in range(1,6):
##        print(counts[i])
    print("Napake:")
    for i in range(len(counts)):
        if counts[i][0] > 0:
            print(counts[i])
    return 



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




if __name__== "__main__":
    X,header = read()
    counts = count(X, header)
    showStats(counts)
    


