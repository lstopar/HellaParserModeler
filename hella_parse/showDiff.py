with open("test_all","r",encoding="utf-8") as f:
    a = list(f.readlines())
with open("test_all_1","r",encoding="utf-8") as f:
    b = list(f.readlines())

for i in range(len(a)):
    a[i] = eval(a[i])
    b[i] = eval(b[i])
    if not sum([a[i][j] == b[i][j] for j in range(len(a[i]))]) == len(a[i]):
        print(sum([a[i][j] == b[i][j] for j in range(len(a[i]))]), len(a[i]))
        print(a[i])
        print(b[i])
        break
        print(i, "Drugacen")
