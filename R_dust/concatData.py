import datetime

file = "result"
outFile = "results03"
interval = datetime.timedelta(0.3)


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

f = open(file, "r")



store = []
startDate = datetime.datetime(1,1,1,0,0,0)
noLines = 0
with open(outFile, "w") as g:
    i = 0
    for line in f.readlines():
        i += 1
        if i == 1:
            g.write(line)
            header = mySplit(line)
            store = [0] * len(header)
        else:
            line = mySplit(line[:-1])
            store = [store[j] + int(line[j]) for j in range(2,len(header)-2)]
            store = [line[0]] + [line[1]] + store + [line[-1]] + [line[-2]]
            noLines += 1
            date = datetime.datetime.fromtimestamp(int(float(store[1])))
            if date - startDate > interval:
                for j in range(2,len(store)-2):
                    store[j] = store[j] if "Bin" in header[j] else store[j] / noLines
                store = [str(j) for j in store]
                g.write(",".join(store))
                g.write("\n")
                store = [0] * len(header)
                noLines = 0
                startDate = date


f.close()