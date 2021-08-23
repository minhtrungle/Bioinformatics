# Import the needed libraries
from matplotlib import style
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import KFold
np.random.seed(42)
style.use('fivethirtyeight')


def prepare_data(factor, feature_type, value_type):
    if (feature_type == "AAC"):
        X = pd.read_csv(
            "../FeatureExtraction/AAC/AAC_data.csv", index_col=0)
        print("../FeatureExtraction/AAC/AAC_data.csv")
    elif (feature_type == "PseAAC"):
        X = pd.read_csv(
            "../FeatureExtraction/PseAAC/PseAAC_data_lambda_{}.csv".format(value_type), index_col=0)
        print("../FeatureExtraction/PseAAC/PseAAC_data_lambda_{}.csv".format(value_type))
    elif (feature_type == "GGAP"):
        X = pd.read_csv(
            "../FeatureExtraction/GGAP/GGAP_data_gap_{}.csv".format(value_type), index_col=0)
        print("../FeatureExtraction/GGAP/GGAP_data_gap_{}.csv".format(value_type))
    else:
        assert False, "Feature path is not exist"

    y = pd.read_csv("./target_data.csv", index_col=0)
    reps = [factor if val == 1 else 1 for val in y.target]
    X = X.loc[np.repeat(X.index.values, reps)]
    y = y.loc[np.repeat(y.index.values, reps)]
    return np.array(X), np.array(y)

def train_test(feature_type, value_type):

    X, y = prepare_data(2, feature_type, value_type)
    kf = KFold(n_splits=10, shuffle=True)
    print(X.shape, y.shape)
    kf.get_n_splits(X)
    sn = [] 
    sp = []
    ACC = []
    MCC = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Set the random state for reproducibility
        fit_rf = RandomForestClassifier(n_estimators=500)
        fit_rf.fit(X_train, y_train.ravel())
        # joblib.dump(fit_rf, "./random_forest.joblib")
        y_pred = fit_rf.predict(X_test)

        print("\n===============")


        joblib.dump(fit_rf, "./random_forest_%s.joblib"%str(accuracy_score(y_test, y_pred)))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        print("SN:",  tp /(tp + fn))
        sn.append(tp /(tp + fn))

        print("SP:", tn / (tn + fp))
        sp.append(tn / (tn + fp))

        print("ACC:", accuracy_score(y_test, y_pred))
        ACC.append(accuracy_score(y_test, y_pred))

        print("MCC:", matthews_corrcoef(y_test, y_pred))
        MCC.append(matthews_corrcoef(y_test, y_pred))

    print("\n Feature: {} | value : {}".format(feature_type, value_type))
    print("SN: min = {:.4f} & max =  {:.4f}".format(np.min(sn), np.max(sn)))
    print("SP: min = {:.4f} & max =  {:.4f}".format(np.min(sp), np.max(sp)))
    print("ACC: min = {:.4f} & max =  {:.4f}".format(np.min(ACC), np.max(ACC)))
    print("MCC: min = {:.4f} & max =  {:.4f}".format(np.min(MCC), np.max(MCC)))

    with open("summary_ectract.txt", "a") as f:
        f.write("\n Feature: {} | value : {}".format(feature_type, value_type))
        f.write("SN: min = {:.4f} & max =  {:.4f}".format(np.min(sn), np.max(sn)))
        f.write("SP: min = {:.4f} & max =  {:.4f}".format(np.min(sp), np.max(sp)))
        f.write("ACC: min = {:.4f} & max =  {:.4f}".format(np.min(ACC), np.max(ACC)))
        f.write("MCC: min = {:.4f} & max =  {:.4f}".format(np.min(MCC), np.max(MCC)))
    return np.max(sn), np.max(sp), np.max(ACC), np.max(MCC)

if __name__ == "__main__":
    over_sn = [] 
    over_sp = []
    over_ACC = []
    over_MCC = []

    ret_sn, ret_sp, ret_acc, ret_mcc = train_test("AAC", None)
    over_sn.append(ret_sn)
    over_sp.append(ret_sp)
    over_ACC.append(ret_acc)
    over_MCC.append(ret_mcc)

    for i in range(1, 21):
        ret_sn, ret_sp, ret_acc, ret_mcc = train_test("PseAAC", i)
        over_sn.append(ret_sn)
        over_sp.append(ret_sp)
        over_ACC.append(ret_acc)
        over_MCC.append(ret_mcc)

    for i in range(0, 20):
        ret_sn, ret_sp, ret_acc, ret_mcc = train_test("GGAP", i)
        over_sn.append(ret_sn)
        over_sp.append(ret_sp)
        over_ACC.append(ret_acc)
        over_MCC.append(ret_mcc)

    with open("summary_ectract.txt", "a") as f:
        f.write("\n Overall")
        f.write("SN: min = {:.4f} & max =  {:.4f}".format(np.min(over_sn), np.max(over_sn)))
        f.write("SP: min = {:.4f} & max =  {:.4f}".format(np.min(over_sp), np.max(over_sp)))
        f.write("ACC: min = {:.4f} & max =  {:.4f}".format(np.min(over_ACC), np.max(over_ACC)))
        f.write("MCC: min = {:.4f} & max =  {:.4f}".format(np.min(over_MCC), np.max(over_MCC)))

