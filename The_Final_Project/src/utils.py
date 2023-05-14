import csv

def read_csv_train_test_acc(file_name):
    with open(file_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        rst_list = []
        epoch_num_count = 0
        for row in csvreader:
            # limit every num in list to 3 digits
            for i in row:
                rst_list.append(round(float(i), 3))
            epoch_num_count += len(row)
        return rst_list, epoch_num_count
    
def read_csv_each_label_acc(file_name, label_num:int):
    with open(file_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        rst_list = [[] for _ in range(label_num)]
        epoch_num_count = 0
        for row in csvreader:
            # limit every num in list to 3 digits
            for i in range(len(row)):
                rst_list[i].append(round(float(row[i]), 3))
            epoch_num_count += 1
        print("rst_list",rst_list)
        print("epoch_num", epoch_num_count)
        return rst_list, epoch_num_count

def record_to_csv(csv_file_path, data):
    # open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        print("data", data)
        writer.writerow(data)
    print('Data written successfully to the CSV file.')

def record_to_csv2d(csv_file_path, data):
    # open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        print("data", data)
        for row in data:
            writer.writerow(row)
    print('Data written successfully to the CSV file.')