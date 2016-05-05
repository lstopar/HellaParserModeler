sensors = {}
# file = "all_data.csv"
file = "outfile.csv"

with open(file,"r") as f:
    i = 0
    for line in f:
        i += 1
        line = line.split(",")
##        if line[1] not in ["name", "Temperature", "Flow_rate",
##                           "IRtemp", "Sampling_period", "Humidity"]:
##            print(line)
##            break
        if (line[0][:7] == "2016-02" and int(line[0][8:10])>18) or line[0][:7] == "2016-03":
            sensors[line[1]] = sensors.get(line[1],0) + 1
        if i % 1000000 == 0:
            print(i)

print(sensors)


##results all:
r1 = {'Bin7(3um)': 652138, 'Bin15(16um-17um)': 652138, 'Bin3(1um)': 652137,
 'Sampling_period': 465990, 'Bin1(0': 652137, 'Bin14(14um)': 652138,
 'Bin12(10um)': 652138, 'Humidity': 7029717, 'Bin11(8um)': 652138,
 'Temperature': 7029747, 'Bin6(2': 652138, 'Bin8(4um)': 652138,
 'Bin10(6': 652138, 'Bin9(5um)': 652138, 'IRtemp': 4216652,
 'Flow_rate': 465990, 'name': 1, 'Bin13(12um)': 652138, 'Bin5(1': 652138,
 'Bin0(0': 652138, 'Bin2(0': 652137, 'Bin4(1': 652138}

##for i,j in r1.items():
##    print(i)

##results 2015-10-30:
r2 = {'Bin5(1': 325754, 'Bin2(0': 325753, 'Bin11(8um)': 325754,
 'Bin3(1um)': 325753, 'Flow_rate': 325754, 'Bin6(2': 325754,
 'Bin10(6': 325754, 'IRtemp': 2160577, 'name': 1, 'Bin8(4um)': 325754,
 'Bin7(3um)': 325754, 'Bin14(14um)': 325754, 'Bin12(10um)': 325754,
 'Bin9(5um)': 325754, 'Bin4(1': 325754, 'Bin13(12um)': 325754,
 'Bin0(0': 325754, 'Bin1(0': 325753, 'Sampling_period': 325754,
 'Humidity': 1319255, 'Bin15(16um-17um)': 325754, 'Temperature': 1319262}

##results 2015-10-30 -> 2015-12-11 08:07:05
r3 = {'IRtemp': 419798, 'Temperature': 410628, 'name': 1, 'Humidity': 410622}
