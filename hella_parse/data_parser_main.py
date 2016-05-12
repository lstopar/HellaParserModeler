import datetime
from os import listdir
from pptx_xml_parser import parseXmlFiles
import pickle
import os
from hella_parse.scrap_parser import parseCsvFiles, get_scrap_types
import csv
from hella_parse.pptx_xml_parser import get_shift
from _datetime import timedelta

dir_name = '/mnt/raidM2T/data/Hella/scrap-data/all/'
readings_path = dir_name + 'readings/'
plan_path = dir_name + 'plan_dela/xml/'
scrap_path = dir_name + 'scrap/'
fout_name = dir_name + 'Hella molding - 2016-05-11.csv'

plan_to_scrap_h = {
    'nova Insignia': 'Insignia Nova - 187.552-01/02',
    'Mercedes 413/414': 'Daimler E-clase BR212 - 166.413/414-00',
    'Edison': 'Edison X07 - 190.890-01/02',
    'Corsa 1': 'Corsa - 195.429-01/02 or.1',
    'Corsa 2': 'Corsa - 195.429-01/02 or.2',
    'X82': 'Renault X82 - 188.709-01/02',
    'Edison': 'Edison X07 - 190.890-01/02',
    'HFE': 'Renault HFE 196.197-01/02',
    'Mercedes 166.071': 'Daimler CLK BR207 - 166.071-01/02',
    'Picasso': 'Picasso PSA 010 - 162.979/980-00',
    'Omega': 'Omega / Cadillac 194.856-01/02',
    'Delta': ['Astra Nova (Delta) GVC 011 - 171.572-01/02', 'Astra Nova (Delta) GVC 010 - 171.572-01/02'],
    'stara Astra': ['Astra 3300-3317 012 - 160.201/202-00', 'Astra 3300-3317 011 - 160.201/202-00'],
    'X10': 'Renault X10 - 180.585-01/02',
    'X12': 'Nissan X12 - 187.829-01/02',
    'Picasso': 'Picasso PSA 010 - 162.979/980-00',
    'BMW 155.031-00 samo levi': 'BMW E 65 -155.031/032-00'
}

def get_timestamp(d):
    return int((d - datetime.datetime(1970,1,1)).total_seconds()) * 1000

def parse(machine = "61282649"):

    #########################################################################
    ## izhodna datoteka

##    ## leta, meseci, dnevi, ki jih preberemo iz filov. Manjkajoce izpusti.
##    config = ([2015],
##               {2015:[11],2016:[1]},
##               {10:31, 11:27, 12:31, 1:31})
    #########################################################################

    product_plans = parseXmlFiles(plan_path)
    
    print(str(product_plans))
    
    scrap_reports = None
    scrap_fname = dir_name + 'scrap_reports-fixdates-rates.p'
    if os.path.isfile(scrap_fname):
        pin = open(scrap_fname, 'rb')
        scrap_reports = pickle.load(pin)
        pin.close()
    else:
        scrap_reports = parseCsvFiles(showMissing=True, path=scrap_path)
        pout = open(scrap_fname, 'wb')
        pickle.dump(scrap_reports, pout)
        pout.flush()
        pout.close()
        
    scrap_types = get_scrap_types()
    
    header_offset = 5
    row_offset = header_offset + 3
    col_offset = 2

    ## izmene. II: 6-14h, III: 14-22h, I: 22-6h
    ## (v planu dela je to edino smiselno)

    header_row = None
    result = []

    files = listdir(readings_path)
    for file in files:
        
        if not file.lower().endswith('csv'):
            continue
        
        print(file)
        
        fin = open(readings_path + file, 'r')
        reader = csv.reader(fin, delimiter=',', quotechar='"')
        rows = [row for row in reader]
        
        n_rows = len(rows)
        n_cols = len(rows[0])
        
        from_to_cell = rows[3][0]
        from_to_str = from_to_cell
        from_to_spl = from_to_str.split(' - ')
        
        start_time = datetime.datetime.strptime(from_to_spl[0], "%d.%m.%y %H:%M")
        end_time = datetime.datetime.strptime(from_to_spl[1], "%d.%m.%y %H:%M")
        
        print('Parsing readings from ' + str(start_time) + ' to ' + str(end_time))
        
        if header_row is None:
                
            header_row = ['date', 'product', 'shift', 'timestamp', 'total parts', 'total scrap'] + scrap_types
            header_row.append('total scrap (%)')
            for scrap_type in scrap_types:
                header_row.append(scrap_type + ' (%)')
                    
            for col_n in range(col_offset, n_cols):
                h0 = rows[header_offset][col_n]
                h1 = rows[header_offset+1][col_n]
                h2 = rows[header_offset+2][col_n]
                
                if h0 is None:
                    h0 = ''
                if h1 is None:
                    h1 = ''
                if h2 is None:
                    break
                    
                header_row.append(h0 + h1 + h2)
        
        for row_n in range(row_offset, n_rows):
            out_row = []
            
            date_str = rows[row_n][0]
            time_str = rows[row_n][1]
            
            if date_str is None:
                break
            
            if date_str == 'Nastavljena vrednost':
                row_n += 1
                continue
            
            reading_time = datetime.datetime.strptime(date_str + ' ' + time_str, "%d.%m.%y %H:%M:%S")
            hour = reading_time.hour
            timestamp = get_timestamp(reading_time)
            
            plan_date_str = None
            if hour < 6:
                plan_date_str = (reading_time - timedelta(days=1)).strftime('%d.%m.%Y')
            else:
                plan_date_str = reading_time.strftime('%d.%m.%Y')
            
            shift_n = get_shift(reading_time)
            
            if not plan_date_str in product_plans:
                print('Don\'t have a plan for date: ' + plan_date_str)
                continue
            
            product_plan = product_plans[plan_date_str]   
            
            # get the correct product
            
            possible_products = product_plan[shift_n]
            product = None
            for product_conf in possible_products:
                product_end_hour = product_conf['end']
                
                if shift_n == 3:
                    if hour <= 23:
                        if hour <= product_end_hour or product_end_hour <= 6:
                            product = product_conf['product']
                            break
                    elif hour < product_end_hour:
                        product = product_conf['product']
                        break
                elif hour < product_end_hour:
                    product = product_conf['product']
                    break
            
            product = product.strip()
            
            if product == 'prazno':
                continue    # TODO can I do anything better here???
            if product is None:
                raise ValueError('Unable to find product! Date: ' + plan_date_str)
            if not product in plan_to_scrap_h:
                raise ValueError('Product ' + product + ' not in hash! Date: ' + plan_date_str)
            
            if not plan_date_str in scrap_reports:
                print('Scrap report for date ' + plan_date_str + ' missing!')
                continue
            
            # extract the scrap report
            scrap_report = scrap_reports[plan_date_str]
            product_scrap = plan_to_scrap_h[product]
            
            if product_scrap is None:
                raise ValueError('Product ' + product + ' not found!')
            
            if isinstance(product_scrap, list):
                for ps in product_scrap:
                    if ps in scrap_report[shift_n]:
                        product_scrap = ps
                        break
            
            out_row.append(date_str + ' ' + time_str)
            out_row.append(product)
            out_row.append(shift_n)
            out_row.append(timestamp)
            
            if not product_scrap in scrap_report[shift_n]:
                print('Scrap report missing for car: ' + product_scrap + '! Date: ' + plan_date_str)
                continue
            
            scraps = scrap_report[shift_n][product_scrap]
            good_parts = scraps['good_parts']
            total_scrap = 0
            for scrap_type in scrap_types:
                total_scrap += scraps[scrap_type]
                
            total_parts = good_parts + total_scrap
            
            out_row.append(total_parts)
            out_row.append(total_scrap)
            for scrap_type in scrap_types:
                out_row.append(scraps[scrap_type])
            
            out_row.append(float(total_scrap) / total_parts if total_parts != 0 else 0)
            for scrap_type in scrap_types:
                out_row.append(float(scraps[scrap_type]) / total_parts if total_parts != 0 else 0)
            
            for col_n in range(col_offset, n_cols):
                value = rows[row_n][col_n]
                
                if value is None or value == '':
                    value = 0
                else:
                    value = float(value)
                
                out_row.append(value)

            result.append(out_row)
        fin.close()
        
    print('Removing duplicates ...')
    timestamp_col = 3
    timestamp_h = { row[timestamp_col]: row for row in result }
    result = [timestamp_h[key] for key in timestamp_h]
    print('Sorting ...')
    result.sort(key = lambda row: row[timestamp_col])
        
    print('Writing to output file ...')
    fout = open(fout_name, 'w')
    fout.write(','.join(['"' + str(val) + '"' for val in header_row]))
    for row_n, out_row in enumerate(result):
        line = ','.join([str(val) for val in out_row])
        fout.write('\n' + line)
            
    fout.flush()
    fout.close()
    
    print('Done!')

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
    machine = "61282649"
    parse(machine)







