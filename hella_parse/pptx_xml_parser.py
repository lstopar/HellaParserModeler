from os import listdir
from _datetime import datetime

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

def get_shift_start(shift_id):
    if shift_id == 1:
        return 6
    elif shift_id == 2:
        return 14
    else:
        return 22
    
def get_shift_end(shift_id):
    if shift_id == 1:
        return 14
    elif shift_id == 2:
        return 22
    else:
        return 6
    
def get_shift(tm):
    hour = tm.hour
    if 22 <= hour or hour < 6:
        return 3
    elif 6 <= hour and hour < 14:
        return 1
    elif 14 <= hour and hour < 22:
        return 2

def parseXmlFiles(
               path="./xmlji",
               show = False,
               showMissing = False):
    """Sparsa vse xml file ppt-ja na poti 'path', ki imajo ime enako
        'dan.mesec.leto.xml' in so podani z 'leta', 'meseci' in 'dnevi'."""
    
    print('Parsing scrap ...')
    
    result = {}
    
    g = open('test_scrap', "w")

    files = listdir(path)
    for file in files:
        print(file)
        
        if not file.endswith('.xml'):
            continue
        
        finish = False
        
        fname_arr = file.split(sep='.')
        day = int(fname_arr[0])
        month = int(fname_arr[1])
        year = int(fname_arr[2])
        
        try:
            tekstovnaPolja = {}
            with open(path + file, 'r') as f:
                i = 0
                data = ""
                for line in f.readlines():
                    i += 1
                    data = line
                    if i == 3:
                        break ## hocemo samo 3 vrstico
                data = data.split("a:p>")

                ## vsak druge element bo znotraj znacke
                tagOpen = False
                stPolja = 0
                prejsnja = False
                for polje in data:
                    if finish: break
                    if not tagOpen:
                        tagOpen = True
                        continue
                    tagOpen = False
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
                #tekstovnaPolja["datum"] = [day, month, year]
                result[datetime(year, month, day).strftime('%d.%m.%Y')] = tekstovnaPolja
                g.write(str(tekstovnaPolja))
                g.write("\n")
        except IOError:
            print('Exception while opening file: ' + file)

    g.close()
    
    # the ID of our machine is 61282649 KM 1000/1
    offset = 8
    result1 = {}
    for date_str in result:
        result1[date_str] = {}
        fields = result[date_str]
        for shift_num in range(3):
            product = fields[offset + shift_num]
            shift_id = shift_num + 1
            
            if '/' in product:
                # we have two products
                products = product.split(' / ')
                # check if the products appear in the description
                
                shift_start = get_shift_start(shift_id)
                shift_end = get_shift_end(shift_id)
                
                if shift_end == get_shift_end(3):
                    shift_end += 24
                
                interval = float(shift_end - shift_start) / len(products)
                
                out = []
                for i, product in enumerate(products):
                    out.append({
                        'product': product,
                        'start': (shift_start + i*interval) % 24,
                        'end': (shift_start + (i+1)*interval) % 24
                    })
                result1[date_str][shift_id] = out
            else:
                result1[date_str][shift_id] = [{
                    'product': product,
                    'start': get_shift_start(shift_id),
                    'end': get_shift_end(shift_id)                           
                }]
            
    
    return result1




        
if __name__ == "__main__":
    parseXmlFiles(show = True, showMissing = True)









































