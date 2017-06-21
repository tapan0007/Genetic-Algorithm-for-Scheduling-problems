import os

SCHEDULER = ["neuralnet"]

DIRECTORY="/storage/others/tapan/gpgpu-simDefault/gpgpu-sim/ispass2009-benchmarks"
NUM_TO_RUN = 1
KERNEL = ["AES"]

def runKernel(benchmarkName, scheduler):
    direc = DIRECTORY + '/' + benchmarkName
    os.chdir(direc)
    print("Running the kernel: " + benchmarkName)
    os.system("bash rungpgpusim " + scheduler + " >> " + "terminallog.txt")
    return

def parseFile(benchmarkName):
    direc = DIRECTORY + '/' + benchmarkName + '/gpgpusim.log'
    file  = open(direc, 'r')
    readLine = file.readlines()
    resultList = []
    #print(readLine)
    for lines in readLine:
        if "gpu_sim_cycle" in lines:
            #print(lines)
            resultList.append((int(lines.split(' ')[-1].strip('\n')), lines.split(' ')[0].strip('\n')))
    if resultList == []:
        print("Error in parsing file")
    file.close()
    return resultList

def outputlog(benchmarkName, scheduler, index, file1):
    print("Save the log")
    getres = parseFile(benchmarkName)
    #file1 = open(benchmarkName + '_' + scheduler + '.txt', 'w')
    file1.write("Index " + str(index)+ " :\n")
    for (res, kern) in getres:
        file1.write(kern + ' : ' + str(res) + '\n')
        print(str(index) + " : " + kern + ' : ' + str(res) + '\n')
    file1.write('\n#########################\n\n')
    return getres[0][0]

if __name__ == "__main__":
    file1 = open(kern + '_' + "NN" + '.txt', 'wb')
    runKernel(kern, SCHEDULER[sch])
    x = outputlog("AES", "neuralnet", index, file1)
    file1.close()


