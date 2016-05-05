f = open("test_scrap","r")
napake = set()
for line in f.readlines():
    line = eval(line[:-1])
    for masina,lin in line.items():
        if masina != "datum":
            for izmena,li in lin.items():
                for napaka,val in li.items():
                    napake.add(napaka)

f.close()
# print(napake)
for i in napake:
    print(i)
