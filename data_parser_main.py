import datetime
from pptx_xml_parser import parseXmlFiles
from slabi_deli_parser import parseCsvFiles


def parse(izbranStroj = "61282649", path = "./masina",
          config = ([2015],{2015:[11],2016:[1]},{10:31, 11:27, 12:31, 1:31}) ):

    #########################################################################

    ## id stroja, kot je zapisan v planu dela (brez /001).za i-kas napisi i-KAS
##    izbranStroj = "61282649"

    ## pot do .csv podatkov. imena filov morajo biti 1, 2, ..., po vrsti.
##    path = "./masina"

    ## izhodna datoteka
    resultFile = "./result_{0}.csv".format(izbranStroj)

##    ## leta, meseci, dnevi, ki jih preberemo iz filov. Manjkajoce izpusti.
##    config = ([2015],
##               {2015:[11],2016:[1]},
##               {10:31, 11:27, 12:31, 1:31})



    #########################################################################



    planDela = parseXmlFiles(*config, showMissing=True)
    slabiDeli = parseCsvFiles(*config, showMissing=True)

    stFila = 0

    ## izmene. II: 6-14h, III: 14-22h, I: 22-6h
    ## (v planu dela je to edino smiselno)

    g = open(resultFile, "w", encoding = "utf-8")

    while True:
        stFila += 1
        file = "{0}/{1}.csv".format(path, stFila)
        try:
            with open(file, "r") as f:
                i = 0
                oldDate = 0
                oldDate1 = 0
                missingTag = 0
                missingDate = 0
                for line in f.readlines():
                    i += 1
                    if not line:
                        continue ## prazna, verjetno zadnja vrstica
                    line = line[:-1].split(",")
                    if i in [1,3,5]:
                        continue
                    if i == 2:
                        naslov = line[0]
                    elif i == 4:
                        daterange = line[0]
                    elif i == 6:
                        header1 = line
                    elif i == 7:
                        header2 = line
                    elif i == 8:
                        header3 = line
                        header3 = ["{0}({1}){2}".format(header1[i],
                                                        header2[i],
                                                        header3[i])
                                   for i in range(len(header3))]
                        header3[1] = "Timestamp"
                        header3[0] = "HumanTimestamp"
                        header3[2] = "Cikel"
                        if stFila == 1:
                            g.write(",".join(header3))
                            ## write še vse ostale headerje iz excela
                            ## vsi produkti/izneme imajo iste napake (vsak dan),
                            ## zato vzamemo kar iz prvega polja
                            header4 = []
                            for key,val in slabiDeli[0].items():
                                if not key == "datum":
                                    pomo = val
                                    break
                            for key in pomo["izmena I"]:
                                header4.append(key)
                            header4.append("StDelavcev")
                            header4.append("Komentar")
                            if header4:
                                g.write(",")
                            g.write(",".join(header4))
                            g.write("\n")
                            header5 = header3 + header4
                            headerMap = {}
                            for k in range(len(header5)):
                                headerMap[header5[k]] = k 
                    elif line[0] == "Nastavljena vrednost":
                        ## todo -> to oni nastavijo na roke??? kako to handlam ???
                        ## todo v odpadnih kosih je xy vrednost.
                        ## verjetno izpustim ???
                        continue
                    else:
                        date = line[0].split(".")
                        time = line[1].split(":")
                        timestamp = datetime.datetime(2000+int(date[2]),
                                                      int(date[1]),
                                                      int(date[0]),
                                                      int(time[0]),
                                                      int(time[1]),
                                                      int(time[2]))
                        line = line[1:]
                        ## todo hocemo tu unix timestamp ???
                        line[0] = str(timestamp)
                        line[1] = str(timestamp.timestamp())
                        ## dodamo podatke iz excela
                        date[2] = 2000 + int(date[2])
                        date = [int(k) for k in date]

                        ura = int(time[0])
                        if 6 <= ura and ura <= 13:
                            izmena = 1
                            imeIzmene = "izmena I"
                        elif 14 <= ura and ura <= 21:
                            izmena = 2
                            imeIzmene = "izmena II"
                        elif ura <= 5 and ura >= 0:
                            izmena = 0
                            imeIzmene = "izmena III" ## gledamo za en dan nazaj
                        elif ura >= 22 and ura <= 23:
                            izmena = 3
                            imeIzmene = "izmena III"
                        else:
                            print("Ura {0} nepravilna".format(ura))
                            raise
                        
                        if not (oldDate == date):
                            ## todo se da optimizirat, ampak se zda zdaj
                            ## ne splaca
                            for dan1 in planDela:
                                date2 = dan1["datum"]
                                if date == date2:
                                    dan = dan1
                                    break
                            if not (date == date2):
                                continue ## ni podatkov
                            oldDate = date
                        ## produkti v izmenah I,II,III,I
                        ## (I je iz podatkov ocitno nocna)
                        if izbranStroj in dan[7]:
                            produkti = [dan[k] for k in range(8,12)]
                        elif izbranStroj in dan[13]:
                            produkti = [dan[k] for k in range(14,18)]
                        elif izbranStroj in dan[19]:
                            produkti = [dan[k] for k in range(20,24)]
                        elif izbranStroj in dan[25]:
                            produkti = [dan[k] for k in range(26,30)]
                        elif izbranStroj in dan[31]:
                            produkti = [dan[k] for k in range(32,36)]
                        elif izbranStroj in dan[36]:
                            produkti = [dan[k] for k in range(37,41)]
                        else:
                            print("Stroj {0} ne obstaja!".format(izbranStroj))
                            raise
                        produkt = produkti[izmena]
                        if not isValid(produkt):
                            continue ## ni produkta, torej ni podatkov
                        try:
                            stDelavcev = dan[42 + izmena]
                        except:
                            print(len(dan), izmena)
                            print(dan)
                            raise
                        
                        ## todo produkti niso natancno zapisani. sparsaj,
                        ## upostevaj zapise spodaj.
                        k = 46
                        dodatniZapisi = []
                        while dan.get(k):
                            dodatniZapisi.append(dan[k])
                            k += 1
                        
                        ## ce pademo v 0-to izmeno moramo gledat datum
                        ## za prejsnji dan, ker so podatki o odmetu tam
                        if izmena == 0:
                            date = substractOne(date)
                        if not (oldDate1 == date):
                            ## todo se da optimizirat, ampak se za zdaj
                            ## ne splaca
                            for dan1 in slabiDeli:
                                date2 = dan1["datum"]
                                if date == date2:
                                    dan2 = dan1
                                    break
                            if not (date == date2):
                                continue ## ni podatkov
                            oldDate1 = date

                        ## todo handlaj deljene produkte v izmeni
                        for key in dan2.keys():
                            ## potrebno, ker so cudni zapisi produktov
                            if isSame(produkt, key, dodatniZapisi, ura, izmena):
                                pomo = key
                                break
                        if (not isSame(produkt, pomo, dodatniZapisi, ura, izmena)
                            and missingTag != produkt
                            and missingDate != dan2["datum"]):
                            print("""Nisem našel ujemanja za produkt(e) {0} dne
{1} (verjetno ni podatkov v excelu)!""".format(
                                produkt,".".join(
                                    [str(k) for k in dan2["datum"]])))
                            missingTag = produkt
                            missingDate = dan2["datum"]
                            ## ne vrnemo napake, ker obstajajo napake v
                            ## danih podatkih.
                            ## raise 
                        try:
                            for k,v in dan2[pomo][imeIzmene].items():
                                ## sinhronizira s headerjem, ker ni nujno isti
                                ## vrstni red iteriranja
                                line = line + [0] * (len(header5) - len(line) + 1)
                                line[headerMap[k]] = v
                        except:
                            print(dan2[pomo])
                            raise
                        line[-2] = stDelavcev
                        line[-1] = '"{0}"'.format(str(";".join(dodatniZapisi)))

                        line = [str(k) for k in line]
                        g.write(",".join(line))
                        g.write("\n")
         
        except IOError:
            print("""\nFile {0} ne obstaja/se ga ne da prebrati!
Ce je bil prejsni file zadnji, sem koncal.""".format(file))
            break

    g.close()
    return

def isValid(produkt):
    produkt = produkt.strip().lower()
    if "ne dela" in produkt:
        return False
    elif produkt == "":
        return False
    return True

def isSame(produkt, key, dodatniZapisi, ura, izmena):
    if key == "datum":
        return False
    produkt = produkt.strip().lower()
    key = key.lower()
    
    if "/" not in produkt:
        produkt = produkt.split(" ")
        for beseda in produkt:
            if beseda == "stara" and "nova" not in key:
                continue
            elif beseda not in key:
                return False
    else:
        ## todo. sedaj razpolovi izmeno, ce se deli
        ## probaj kaj sparsat
        produkt = produkt.split("/")
        produkt[0] = produkt[0].strip().split(" ")
        produkt[1] = produkt[1].strip().split(" ")
        if len(produkt) > 2:
            print("Veckratna menjava produkta! {0}".format(produkt))
            raise
        if izmena == 0:
            polovica = 0
        elif izmena == 1:
            polovica = 10
        elif izmena == 2:
            polovica = 18
        elif izmena == 3:
            polovica = 0

        ## na roke prebrano:
        if produkt[0] == ['nova', 'insignia'] and produkt[1] == ['x12']:
            polovica = 1
        elif produkt[0] == ['x12']:
            polovica = 18
        elif produkt[0] == ['edison']:
            polovica = 8
        elif produkt[0] == ['picasso']:
            polovica = 10
        elif produkt[0] == ['x10']:
            polovica = 22
        elif produkt == ['stara', 'astra']:
            polovica = 4
        elif produkt[0] == ['nova', 'insignia']and produkt[1] == ['x10']:
            polovica = 8
        if ura >= polovica:
            produkt = produkt[1]
        else:
            produkt = produkt[0]
        for beseda in produkt:
            if beseda == "stara" and "nova" not in key:
                continue
            elif beseda not in key:
                return False
    return True

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
    


if __name__ == "__main__":
    parse()







