import datetime
import time as ttime
file = "data1.csv"
outFile = "data.csv"
cutBefore = [2016,2,1]

f = open(file, "r")
g = open(outFile, "w")
for line in f.readlines():
    if not line:
        continue
    elif line[:9] == "timestemp":
        line = "HumanTimestamp" + line
        g.write(line)
        continue
    line = line.split(",")
    datetime2 = line[0].split(" ")
    date = datetime2[0].split("-")
    date = [int(i) for i in date]
    time = [int(i) for i in datetime2[1].split(":")]
    if date < cutBefore or date[0] >  2016:
        ## tule so nesmiselni podatki za 2030 notri.
        continue
    line = [line[0]] + line
    g.write(",".join(line)) ## ze ima \n
    
    
g.close()
f.close()
