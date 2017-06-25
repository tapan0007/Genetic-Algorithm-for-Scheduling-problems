import os
import sys
#SCHEDULER = ["neuralnet"]

DIRECTORY="/storage/others/tapan/gpgpu-simDefault/gpgpu-sim/ispass2009-benchmarks"
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
    file1.close()
    return getres[0][0]

if __name__ == "__main__":
    scheduler = "neuralnet_" + sys.argv[1] + "_" + sys.argv[2]
    file1 = open(KERNEL[0] + '_' + "NN" + '.txt', 'w')
    runKernel(KERNEL[0], scheduler)
    fitness = outputlog("AES", "neuralnet", 1, file1)
    os.chdir("/storage/others/tapan/gpgpu-simDefault/gpgpu-sim/v3.x/ANN_DATA/genetic_algorithm")
    file2 = open('data/' + 'generation_' + sys.argv[1] + '/fitness_' + sys.argv[2] + '.txt', 'w')
    file2.write(str(fitness))
    file1.close()
    file2.close()

