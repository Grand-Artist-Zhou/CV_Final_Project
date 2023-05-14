import matplotlib.pyplot as plt
from utils import read_csv_each_label_acc, read_csv_train_test_acc

# draw over all accs
def draw_acc_vs_epoch(epoch_num, acc_train_list, acc_test_list, graph_title):
    # sample data
    x = [i for i in range(1, epoch_num + 1)]

    # create a figure and axis object
    fig, ax = plt.subplots()

    # plot the data as a line graph
    ax.plot(x, acc_train_list, label="train accuracy")
    ax.plot(x, acc_test_list, label="test accuracy")

    # set the axis labels and title
    ax.set_xlabel('epoch nums')
    ax.set_ylabel("accuracy")
    ax.set_title(graph_title)

    plt.legend()

    # display the graph
    plt.savefig('../images/' + graph_title + '.png')


# draw models accs
def draw_acc_two_list_vs_epoch(epoch_num, acc_list1, acc_list2, graph_title):
    # sample data
    x = [i for i in range(1, epoch_num + 1)]

    # create a figure and axis object
    fig, ax = plt.subplots()

    # plot the data as a line graph
    ax.plot(x, acc_list1[1], label=acc_list1[0])
    ax.plot(x, acc_list2[1], label=acc_list2[0])

    # set the axis labels and title
    ax.set_xlabel('epoch nums')
    ax.set_ylabel("accuracy")
    ax.set_title(graph_title)

    plt.legend()

    # display the graph
    plt.savefig('../images/' + graph_title + '.png')

# draw single class accs
def draw_each_label_acc_vs_epoch(epoch_num, each_label_data_list, graph_title):
    # sample data
    x = [i for i in range(1, epoch_num + 1)]
    # print("x",x); print("y",y)

    # create a figure and axis object
    fig, ax = plt.subplots()
    
    # plot the data as a line graph
    label = []
    if len(each_label_data_list) == 4:
        label.extend(["heavy plastic", "no image", "no plastic", "some plastic"])
    else:
        label.extend(["heavy plastic", "no plastic", "some plastic"])
    for i in range(len(each_label_data_list)):
        ax.plot(x, each_label_data_list[i], label=label[i])

    # set the axis labels and title
    ax.set_xlabel('epoch nums')
    ax.set_ylabel('accuracy')
    ax.set_title(graph_title)

    plt.legend()

    # display the graph
    plt.savefig('../images/' + graph_title + '.png')

# driver code only for this time tasks

## draw 3 label relate graphs
file_name = "../data/graph_data/3_train.csv"
train_list, epoch_num_count = read_csv_train_test_acc(file_name)
file_name = "../data/graph_data/3_test.csv"
test_list, epoch_num_count = read_csv_train_test_acc(file_name)
draw_acc_vs_epoch(epoch_num_count, acc_train_list=train_list
                  , acc_test_list=test_list, graph_title="ResNet18 for 3-class Classification Train Test Accuracy")

file_name = "../data/graph_data/3_true_Y.csv"
rtn_list, epoch_num_count = read_csv_each_label_acc(file_name, 3)
draw_each_label_acc_vs_epoch(epoch_num_count, rtn_list, "ResNet18 for 3-class Classification Each Label Accuracy")

## draw 4 label relate graphs
file_name = "../data/graph_data/4_train.csv"
train_list, epoch_num_count = read_csv_train_test_acc(file_name)
file_name = "../data/graph_data/4_test.csv"
test_list, epoch_num_count = read_csv_train_test_acc(file_name)
draw_acc_vs_epoch(epoch_num_count, acc_train_list=train_list
                  , acc_test_list=test_list, graph_title="ResNet18 for 4-class Classification Train Test Accuracy")

file_name = "../data/graph_data/4_true_Y.csv"
rtn_list, epoch_num_count = read_csv_each_label_acc(file_name, 4)
draw_each_label_acc_vs_epoch(epoch_num_count, rtn_list, "ResNet18 for 4-class Classification Each Label Accuracy")

## draw 3/4 train test acc
file_name = "../data/graph_data/3_train.csv"
list1, epoch_num_count = read_csv_train_test_acc(file_name)
file_name = "../data/graph_data/4_train.csv"
list2, epoch_num_count = read_csv_train_test_acc(file_name)
draw_acc_two_list_vs_epoch(epoch_num_count, acc_list1=("3-class train accuracy",list1)
                  , acc_list2=("4-class train accuracy",list2), graph_title="ResNet18 for 3-4-class Classification Train Accuracy")

file_name = "../data/graph_data/3_test.csv"
list1, epoch_num_count = read_csv_train_test_acc(file_name)
file_name = "../data/graph_data/4_test.csv"
list2, epoch_num_count = read_csv_train_test_acc(file_name)
draw_acc_two_list_vs_epoch(epoch_num_count, acc_list1=("3-class test accuracy",list1)
                  , acc_list2=("4-class test accuracy",list2), graph_title="ResNet18 for 3-4-class Classification Test Accuracy")