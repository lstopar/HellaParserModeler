file = "result_61282649"
outFile = "result"
g = open(outFile, "w")
with open (file, "r") as f:
    for line in f.readlines():
        line = line.split(",")
        line = ",".join(line[1:])
        g.write(line)

g.close()