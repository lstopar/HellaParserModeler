file = "result_61282649_dust"
outFile = "result"

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


g = open(outFile, "w")
with open (file, "r") as f:
    i = 0
    for line in f.readlines():
        i += 1
        if i == 1:
            line = mySplit(line)
            line1 = []
            for name in line:
                if '"' in name:
                    name = name[1:-1]
                if "," in name:
                    name = "".join(name.split(","))
                name = name.split("(")[0]
                line1.append(name)
            line = ",".join(line1)
        g.write(line)

g.close()
