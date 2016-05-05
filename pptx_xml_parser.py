########################### Opazke glede pptx formata izvoženega v xml. 

## parsaj samo vrstico 3 xml fila.

## ves text je v tagih "<a:t>"
## polja so "<a:p>", torej združuje tekst znotraj iste znacke.
## v tabeli so a:txBody
## ampak samo, ce tekst obstaja, ker ga je vec
## tagi po vrsti:

########################### Naslov
## Naslov

########################### Tabela
## "I Izmena"
## "Il"
## " izmena"
## "Ill"
## " izmena"
## "I Izmena"
########## Tudi imena produktov so lahko deljena (pri "ne dela" opazil)
## ime mašine 1
## koda mašine 1
## produkt 1
## produkt 2
## produkt 3
## produkt 4
## mašina 2
## koda 2
## produkt 1
## produkt 2 "/"
## produkt 2 del 2 (menjava)
## produkt 3
## produkt 4
## mašina 3
## koda 3
## produkt 1-4
## mašina 4
## koda 4
## produkt 1-2
## produkt 3 (menjava)
## produkt 3 2 del
## produkt 4
## mašina 5
## koda 5
## produkt 1-5
## mašina 6 (ni kode) - mašina = i-KAS
## produkt 1-4
## "št. delavcev"
## št. delavcev v izmeni 1-4 ... kaj je "-1 (T2)" in "-2 (T2)" zraven števil???

############################# Spodnji del
## tekst v vec poljih
## ce konca z ":" oz. rece prioriteta je potem seznam prioritet (1-5)
## tudi tekst lahko razbit med vec polj

## na koncu je "1" in "Mastertitelformat bearbeiten"


## slovar stevil polj:
##1 -> Naslov
##2-5 -> Ime izmene
##6,7 -> Ime, oznaka stroja
##8-11 -> Produkti
##12,13 -> Ime, oznaka stroja
##14-17 -> Produkti
##18,19 -> Ime, oznaka stroja
##20-23 -> Produkti
##24,25 -> Ime, oznaka stroja
##26-29 -> Produkt
##30,31 -> Ime, oznaka stroja
##32-35 -> Produkti
##36 -> Ime stroja
##37-40 -> Produkti
##41 -> "št. delavcev"
##42-45 -> št. delavcev v izmeni
##46-X -> tekst na dnu


def parseXmlFiles(leta=[2015],
               meseci={2015:[11],2016:[1]},
               dnevi={10:31, 11:20, 12:31, 1:31},
               path="./xmlji",
               show = False,
               showMissing = False):
    """Sparsa vse xml file ppt-ja na poti 'path', ki imajo ime enako
        'dan.mesec.leto.xml' in so podani z 'leta', 'meseci' in 'dnevi'."""
    result = []

    for leto in leta:
        mesecov = meseci[leto]
        for mesec in mesecov:
            for dan in range(1,dnevi[mesec]+1):
                finish = False
                file = "{0}/{1}.{2}.{3}.xml".format(path, dan, mesec, leto)
                try:
                    tekstovnaPolja = {}
                    with open(file, "r", encoding = "utf-8") as f:
                        i = 0
                        data = ""
                        for line in f.readlines():
                            i += 1
                            data = line
                            if i == 3:
                                break ## hocemo samo 3 vrstico
                        data = data.split("a:p>")

                        ## vsak druge element bo znotraj znacke
                        odprtaZnackaP = False
                        stPolja = 0
                        prejsnja = False
                        for polje in data:
                            if finish: break
                            if not odprtaZnackaP:
                                odprtaZnackaP = True
                                continue
                            odprtaZnackaP = False
                            toPolje = []
                            if prejsnja:
                                toPolje.append(prejsnja + " ")
                            pol = polje.split("a:t>")
                            if len(pol) < 2:
                                continue
                                ## ni teksta, torej je to odvecen xml konstrukt
                            odprtaZnackaT = False
                            for delec in pol:
                                if not odprtaZnackaT:
                                    odprtaZnackaT = True
                                    continue
                                odprtaZnackaT = False
                                toPolje.append(delec[:-2]) ## brez "</"
                            toPolje = "".join(toPolje)
                            if "<" in toPolje:
                                continue
                            if ("Mastertextformat bearbeiten" in toPolje
                                or "Mastertitelformat bearbeiten" in toPolje
                                or ("1" == toPolje and stPolja > 45)):
                                finish = True
                                continue
                            if prejsnja:
                                prejsnja = False
                            elif " /" in toPolje:
                                prejsnja = toPolje
                                continue
                            stPolja += 1

                            tekstovnaPolja[stPolja] = toPolje
                            
                        if show:
                            print("Dolzina fila {0}: {1}".format(
                                file,
                                len(tekstovnaPolja)))
                        tekstovnaPolja["datum"] = [dan, mesec, leto]
                        result.append(tekstovnaPolja)
                except IOError:
                    if showMissing: print("File {0} ne obstaja.".format(file))
                
    return result




        
if __name__ == "__main__":
    parseXmlFiles(show = True, showMissing = True)









































