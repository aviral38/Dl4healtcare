from collections import defaultdict
import time
import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from Utils.RARE_INFO import RareInfo
from Model.load_dataset import load_diseases_map
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.decomposition import PCA

def evaluation(name, labels, output, topk=(1, 2, 3, 4, 5)):
    print('baseline:', name)
    

    # shape: number of batches * number of classes
    target = labels
    output = output  # shape: batchnum * classnum

    maxk = max(topk)
    batch_size = target.shape[0]

    def partition_arg_topK(matrix, K, axis=-1):
        a_part = np.argpartition(matrix, K, axis=axis)
        if axis == 0:
            row_index = np.arange(matrix.shape[1 - axis])
            a_sec_argsort_K = np.argsort(
                matrix[a_part[0:K, :], row_index], axis=axis)
            return a_part[0:K, :][a_sec_argsort_K, row_index]
        else:
            column_index = np.arange(matrix.shape[1 - axis])[:, None]
            a_sec_argsort_K = np.argsort(
                matrix[column_index, a_part[:, 0:K]], axis=axis)
            return a_part[:, 0:K][column_index, a_sec_argsort_K]

    pred = partition_arg_topK(-output, maxk, axis=1)
    
    
    
    correct = np.zeros((batch_size, maxk))
    for i in range(batch_size):
        for k in range(maxk):
            correct[i, k] = 1 if target[i][pred[i, k]] == 1 else 0

    correct = correct.T
    

    correct_target = target.sum(axis=1)


    for k in topk:
        correct_k = correct[:k].sum(axis=0)
   

        precision_k = 0.0
        recall_k = 0.0
        for i in range(0, batch_size):
   
            _k = k
            precision_k += float(correct_k[i]) / _k
            
            recall_k += float(correct_k[i]) / float(correct_target[i])
        precision_k = precision_k / batch_size
        recall_k = recall_k / batch_size

        f1_k = 2 * precision_k * recall_k / (precision_k + recall_k)

        print("precision @ k=%d : %.5f, recall @ k=%d : %.5f, f1 k=@ %d : %.5f" % (
            k, precision_k, k, recall_k, k, f1_k))


def load_multi_graph(graph_path):
    graph_path='./data/graph_data/191210/graph-P-191210-00'
   
    pos = graph_path.find("/data")

    data_path = graph_path[:pos +5]
    graph_date = graph_path[pos +17: pos + 23]
    graph_type = graph_path[pos + 30]

    map_path = "{}/graph_data/{}/diseases-map-{}.txt".format(
        data_path, graph_date, graph_date)
    

    
    diseases_map = load_diseases_map(map_path)
    
    
    node_list = []
    node_label = {}
    node_attr = {}
    adj_lists = defaultdict(set)
    with open(graph_path + ".node", "r", encoding="utf8") as f:
        for line in f:
            line = line.strip("\n").split("\t")
            if line[0] not in node_list:
                node_list.append(line[0])
            n_label = np.zeros(len(diseases_map))
            if graph_type == 'M':
                n_label[diseases_map[line[0]] - 1] = 1
            if graph_type == 'P':
                n_label[list(
                    map(lambda x: diseases_map[x] - 1, line[1:-2]))] = 1
            main_disease = int(line[-2]) - 1
            rare_flag = int(line[-1])
            node_label[line[0]] = n_label
            node_attr[line[0]] = (main_disease, rare_flag)

    with open(graph_path + ".edge", "r", encoding="utf8") as f:
        for line in f:
            line = line.strip("\n").split("\t")
            adj_lists[line[0]].add(line[1])

    node_map = {}
    feature_map = {}
    for node in node_list:
        node_map[node] = len(node_map)
        for adj in adj_lists[node]:
            if adj not in feature_map:
                feature_map[adj] = len(feature_map)

    feat_data = np.zeros((len(node_map), len(feature_map)), dtype=np.float64)
    labels = np.zeros((len(node_list), len(diseases_map)), dtype=np.int64)
    main_disease = np.zeros((len(node_list), 1), dtype=np.int64)
    rare_type = np.zeros((len(node_list), 1), dtype=np.int64)

    for node in node_list:
        node_id = node_map[node]
        labels[node_id] = node_label[node]
        rare_type[node_id] = node_attr[node][1]
        main_disease[node_id] = node_attr[node][0]
        for neighbor in adj_lists[node]:
            feat_data[node_id, feature_map[neighbor]] = 1

    file_name_train = graph_path + "-transductive-train.index"
    file_name_test = graph_path + "-transductive-test.index"

    with open(file_name_train, "r", encoding="utf8") as f:
        train = [node_map[line.strip("\n")] for line in f]
    with open(file_name_test, "r", encoding="utf8") as f:
        test = [node_map[line.strip("\n")] for line in f]

    multi_test = [i for i in np.where(rare_type > 0)[0].squeeze() if i in test]

    return feature_map, train, test, multi_test, feat_data, labels, main_disease


def run_svm(data_path, file_date, file_suffix):
    data_path = data_path + "/graph_data/" + file_date

    file_patient_graph = data_path + "/graph-P-" + file_date + "-" + file_suffix

    feature_map, train, test, multi_test, feat_data, labels, main_disease = load_multi_graph(
        file_patient_graph)

    # print(feature_map_medical.keys() & feature_map_patient.keys())
    # print(len(feature_map_medical.keys() & feature_map_patient.keys()))

    feat_train = feat_data[train]
    label_train = labels[train]
    feat_test = feat_data[test]
    label_test = labels[test]
    # main_disease_train = main_disease[train]
    # main_disease_test = main_disease[test]

    # feat_test_multi = feat_data[multi_test]
    # label_test_multi = labels[multi_test]
    # main_disease_test_multi = main_disease[multi_test]

  

    test_rare_index = [test.index(i) for i in multi_test]

    clf = svm.SVC(kernel='linear', probability=True)
    
    

    #t_time = time.time()
    clf = clf = OneVsRestClassifier(clf, n_jobs=-1)
    clf.fit(feat_train, label_train)

    #t_time = time.time()
    # output_test = clf.predict(feat_test)
    pred = clf.predict_proba(feat_test)
    
    print()
 

    # overall:
    evaluation('SVM' + ', overall:', label_test, pred)

    # rare:

    evaluation('SVM' + ', rare:', label_test[test_rare_index],
               pred[test_rare_index])


    label_train=np.argmax(label_train, axis=1)
    # w = clf.estimators_[0].coef_[0]
    # a = -w[0] / w[1]
    # xx = np.linspace(np.min(feat_data[:,0]), np.max(feat_data[:,0]), 10)
    # yy = a * xx - (clf.estimators_[0].intercept_[0]) / w[1]

    # # Plot the decision boundary
    # plt.plot(xx, yy, 'k-')

    # # Plot the data points with different colors based on their predicted class
    # Z = clf.decision_function(feat_test)
    # colors = ['r', 'b', 'g'] # Assuming there are three classes
    # for i, color in zip(range(len(colors)), colors):
    #     idx = np.where(label_test == i)[0]
    #     plt.scatter(feat_test[idx, 0], feat_test[idx, 1], c=color, label=i,
    #                 cmap=plt.cm.Paired, edgecolor='black', s=20, alpha=0.8)

    # plt.legend()
    # plt.title('One vs Rest SVM')
    # plt.show()

    # Get support vectors and hyperplane
    
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(feat_train)
    X=X_reduced
    y=label_train
    # Train an SVM classifier
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X, y)

    # Visualize the decision boundary in the 2D projection
    # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Paired)
    # ax = plt.gca()
    # DecisionBoundaryDisplay.from_estimator(
    #     clf,
    #     X_reduced,
    #     ax=ax,
    #     grid_resolution=50,
    #     plot_method="contour",
    #     colors="k",
    #     levels=[-1, 0, 1],
    #     alpha=0.5,
    #     linestyles=["--", "-", "--"],
    # )
    # plt.title("Decision boundary in 2D PCA space")
    # plt.xlabel("PCA component 1")
    # plt.ylabel("PCA component 2")
    # plt.show()


    # plot_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    # for i in range(len(main_disease)):
    #     disease_indices = np.where(labels == i)[0]
    #     if len(disease_indices) == 0:
    #         continue
    #     plot_color = plot_colors[i % len(plot_colors)]
    #     plot_label = main_disease[i]
    #     disease_test = feat_data[disease_indices]
    #     label_test = labels[disease_indices]
    #     disease_proba = clf.predict_proba(disease_test)
    #     disease_fpr, disease_tpr, _ = plot_roc_curve(
    #         label_binarize(label_test, classes=[i, ]),
    #         disease_proba[:, i])
    #     plt.plot(disease_fpr, disease_tpr, color=plot_color,
    #             lw=2, label=plot_label)

    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('One vs Rest ROC curve')
    # plt.legend(loc="lower right")
    # plt.show()





    
    
    

if __name__ == "__main__":
    run_svm("../data", file_date="191210", file_suffix="00")
