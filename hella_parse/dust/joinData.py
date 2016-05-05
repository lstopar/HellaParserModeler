file = "data.csv"
outFile = "out.csv"
f = open(file, "r")
g = open(outFile, "w")
j = 0
used = set()
header = False
lines = []
for line in f.readlines():
    j += 1
    if j == 1:
        continue
    line = line[:-1].split(",")
    if len(line) > 4:
        line = line[:2] + [",".join(line[2:-1])] + [line[-1]]
    if line[2][:3] == "Bin":
        used.add(line[2])
        lines.append(line)
    if len(used) == 16: ## imamo 16 bin-ov
        if not header:
            used1 = list(used)
            headerMap = {}
            for i in used:
                headerMap[i] = used1.index(i) + 2
            used1 = ['"{0}"'.format(i) for i in used1]
            header = ["HumanTimestamp","Timestamp"] + used1
            header = ",".join(header)
            g.write(header)
            g.write("\n")
        
        line = [0] * (len(used1) + 2)
        for i in lines:
            binn = i[2]
            count = i[3]
            line[headerMap[binn]] = count
        line[0] = lines[-1][0]
        line[1] = lines[-1][1]
        
        try:
            g.write(",".join(line))
        except:
            print(headerMap)
            print(line)
            print(lines)
            print(len(lines))
            print(used)
            raise
        g.write("\n")
        used = set()
        lines = []
        
g.close()
f.close()
