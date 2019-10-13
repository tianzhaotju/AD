import csv
import os

data_names = ["cifar10","mnist"]
data_names = ["mnist"]
model_names = ["cae","if","kde","ocsvm","one_class_svdd","soft_boundary_svdd"]
for data_name in data_names:
    w = csv.writer(open(str(data_name)+".csv",'w'))
    for i in range(10):
        result = []
        filenames =[]
        for model_name in model_names:
            filenames.append("./log/%s_%d_%s/log.txt"%(data_name,i,model_name))


        for filename in filenames:
            if not os.path.exists(filename):
                result.append(0)
                continue
            f = open(filename)
            lines = f.readlines()

            for line in lines:
                if "Test AUC" in line:
                    AUC =  float(line.split(":")[1].split("%")[0])
                    result.append(AUC)
                    break
        w.writerow(result)



