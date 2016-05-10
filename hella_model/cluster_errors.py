import re

fname = '/home/lstopar/workspace/JSI/projects/ProaSense/code/HellaPred/HellaModel/docs/scrap-attributes.txt'
fin = open(fname, 'r')

scrap_types = []

for line in fin.readlines():
    if line.startswith('-'):
        scrap_types.append({ 'scrap': line.replace('\n', ''), 'attributes': [] })
        continue
    
    if len(line) == 0 or line[0] == '=' or line[0] == '-' or len(line) < 3:
        continue
    
    scrap_conf = scrap_types[-1]
    attributes = scrap_conf['attributes']
    
    spl = re.split(r'\t+', line)
    attr = spl[0]
    typ = spl[-1]
    
    attributes.append((attr + '-' + typ).replace('\n', '').replace('lowest', 'low').replace('highest', 'high'))
    
fin.close()

print(str(scrap_types))

for scrap_conf in scrap_types:
    scrap_conf['attributes'] = set(scrap_conf['attributes'])
    
for i in range(len(scrap_types)-1):
    for j in range(i+1, len(scrap_types)):
        s0 = scrap_types[i]
        s1 = scrap_types[j]
        
        union = s0['attributes'].union(s1['attributes'])
        intersection = s0['attributes'].intersection(s1['attributes'])
        
        sim = float(len(intersection)) / len(union)
        
        print(s0['scrap'] + '\t-\t' + s1['scrap'] + '\t\t sim: ' + str(sim))