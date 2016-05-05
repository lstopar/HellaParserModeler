f = open("test_all","r",encoding="utf-8")

for line in f.readlines():
    line = eval(line[:-1])
    break    
f.close()
#print(line)
for i in line:
    print(i)
