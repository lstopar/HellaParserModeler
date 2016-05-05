def parseCsvFiles(leta=[2015],
               meseci={2015:[11],2016:[1]},
               dnevi={10:31, 11:27, 12:31, 1:31},
               path="./odpad",
               show = False,
               showMissing = False):
    """Sparsa vse csv file excela-ja na poti 'path', ki imajo ime enako
        'dan.mesec.leto.csv' in so podani z 'leta', 'meseci' in 'dnevi'."""

    result = []

    for leto in leta:
        mesecov = meseci[leto]
        for mesec in mesecov:
            for dan in range(1,dnevi[mesec]+1):
                finish = False
                file = "{0}/{1}.{2}.{3}.csv".format(path, dan, mesec, leto)
                try:
                    polja = {}
                    produkti = []
                    with open(file, "r") as f:
                        i = 0
                        for line in f.readlines():
                            i += 1
                            if not line:
                                continue ## prazna, verejtno zadnja vrstica
                            line = mySplit(line[:-1]) ## brez "\n"
                            if i in [1,2,4,5,6]:
                                continue
                            elif i == 3:
                                k = 4
                                ## imena KAS produktov inicializirana v slovar
                                while line[k]:
                                    polja[line[k]] = {"izmena I":{},
                                                       "izmena II":{},
                                                       "izmena III":{}}
                                    produkti.append((line[k],k))
                                    k += 6
                                ## najdemo i-Kas
                                k += 1
                                while not line[k]:
                                    k += 6
                                ## inicializiramo i-Kas produkte
                                while line[k]:
                                    polja[line[k]] = {"izmena I":{},
                                                       "izmena II":{},
                                                       "izmena III":{}}
                                    produkti.append((line[k],k))
                                    k += 6
                            elif i in range(7,47): ## vkljucno do 46
                                for produkt,k in produkti: 
                                    polja[produkt]["izmena III"][line[3]] = (
                                        doSum(line[k:k+6]))
                            elif i in range(61,101):
                                for produkt,k in produkti: 
                                    polja[produkt]["izmena II"][line[3]] = (
                                        doSum(line[k:k+6]))
                            elif i in range(120,160):
                                for produkt,k in produkti: 
                                    polja[produkt]["izmena I"][line[3]] = (
                                        doSum(line[k:k+6]))
                    polja["datum"] = [dan, mesec, leto]
                    result.append(polja)
                    if show:
                        print("File {0} koncal.".format(file))
                except IOError:
                    if showMissing: print("File {0} ne obstaja.".format(file))
    return result

def doSum(lis):
    try:
        lis = [ int(i) if i and i != ' ' and i != 'xy' else 0 for i in lis]
    except:
        print(lis)
        raise
    return sum(lis)

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
            

if __name__ == "__main__":
    parseCsvFiles(show = True, showMissing = True)
















