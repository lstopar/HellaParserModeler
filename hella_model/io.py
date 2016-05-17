import csv
import numpy as np

def readData(file, target_scrap):
    fin = open(file, "r", encoding = "utf-8")
    
    shift_idx = 2
    scrap_offset = 4
    readings_offset = 50
    
    reader = csv.reader(fin, delimiter=',', quotechar='"')
    rows = [row for row in reader]
    
    headers = rows[0]
    scrap_headers = headers[scrap_offset:readings_offset]
    feature_headers = headers[readings_offset:]
    rows = rows[1:]
    
    shifts = []
    scraps = []
    curr_shift_id = -1
    for row in rows:
        shift_id = row[shift_idx]
        if shift_id != curr_shift_id:
            curr_shift_id = shift_id
            shifts.append([])
            scraps.append([float(val) for val in row[scrap_offset:readings_offset]])
        shift = shifts[-1]
        shift.append([float(val) for val in row[readings_offset:]])
        
    fin.close()
    
    new_shifts = []
    new_scraps = []
    for i, row in enumerate(scraps):
        total_parts = row[0]
        scrap_parts = row[1]
        
        if total_parts < 200 or scrap_parts == 0 or scrap_parts == total_parts:
            print('Omitting shift ' + str(i))
            continue
        
        if total_parts == 490 or total_parts == 580:    # problematic shifts
            print('Omitting shift ' + str(i))
            continue
        
        new_shifts.append(shifts[i])
        new_scraps.append(scraps[i])
    
    return new_shifts, new_scraps, feature_headers, scrap_headers

def preprocess(shifts, ftr_headers, scrap_headers):
    print('Preprocessing ...')

    ftr_headers.append('intersect')
    for shift in shifts:
        for ftrvec in shift:
            ftrvec.append(1)
    
    for shift_n in range(len(shifts)):
        shifts[shift_n] = np.matrix(shifts[shift_n])
        
    return shifts, ftr_headers, scrap_headers

def extract_scraps(scraps, scrap_headers, target_group):
    resp_idxs = [scrap_headers.index(header) for header in target_group]
    resp_arrs = [np.array([row[resp_idx] for row in scraps]) for resp_idx in resp_idxs]
    
    resp = resp_arrs[0]
    for i in range(1, len(resp_arrs)):
        resp = np.add(resp, resp_arrs[i])
        
    return np.array(resp)

n_percentiles = 6

group1 = ['141 zagon (%)', '14 pike, pikasto (razno) (%)', '15 meglica (mat, siva površina) (%)']
ignore1 = []#[51, 89]

group2 = ['144 ponovni zagon (po čiščenju orodja) (%)', '23 nitke (%)', '1 nezalito, nedolito (%)']
ignore2 = []#[112]

group3 = ['5 opraskano (robot, trak) (%)', '7 žarki, plastika (%)', '6 onesnaženo s tujki (%)']
ignore3 = []#[15]

group4 = ['4 počeno, zvito, deformirano (%)', '428 črna(e) pika(e) (%)']
ignore4 = []#[51]

group_single = [
    '14 pike, pikasto (razno) (%)',
    '15 meglica (mat, siva površina) (%)',
    '23 nitke (%)', '1 nezalito, nedolito (%)',
    '5 opraskano (robot, trak) (%)',
    '7 žarki, plastika (%)',
    '6 onesnaženo s tujki (%)',
    '4 počeno, zvito, deformirano (%)',
    '428 črna(e) pika(e) (%)',
    '172 U - zanka (pentlja, hakeljček) (%)',
    '9 enojni žarek (%)'
]
ignore_single = []

group_all = group1 + group2 + group3 + group4 + ['172 U - zanka (pentlja, hakeljček) (%)', '9 enojni žarek (%)']
ignore_all = ignore1 + ignore2 + ignore3 + ignore4