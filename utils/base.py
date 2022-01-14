import os
import numpy as np
import shutil
from pathlib import Path
import pathlib
from PIL import Image
from torchvision.transforms.transforms import RandomAffine, RandomRotation
from utils.preprocessing import (
    get_random_fls,
    mv_files,
    dict_to_df,
    get_file_len,
    get_combined_anomalies,
    get_n_combined_classes,
)
from sklearn.ensemble import IsolationForest
from natsort import os_sorted
import torch
from sklearn.metrics import matthews_corrcoef
from training import training_loop, train, test, test_classifer
from .preprocessing import ImageFolderWithPaths
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score

# from sklearnex import patch_sklearn
from pdb import set_trace

# patch_sklearn()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device_cpu = torch.device("cpu")


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def free_gpu():
    return torch.cuda.empty_cache()


def save_list(my_lst, fl_name):
    np.save(fl_name, my_lst)


def load_lst(fl_name):
    return np.load(fl_name, allow_pickle=True).tolist()


def get_latent_experiment(
    test_pth,
    _transform,
    _test_transform,
    fc1_path,
    fc2_path,
    fc3_path,
    fc4_path,
    batch_size_train,
    batch_size_test,
    n_epoch,
    optim,
    Net,
    learning_rate,
    momentum,
    train,
    saved_model_path,
    anomaly_label,
    num_to_display,
    device,
):
    print("****************Getting Values for Latent Space**********")

    fc1_latent_rws = []
    fc1_latent_mcc = []
    fc1_latent_recall = []
    fc1_latent_precision = []
    fc1_latent_frac = []

    fc2_latent_rws = []
    fc2_latent_mcc = []
    fc2_latent_recall = []
    fc2_latent_precision = []
    fc2_latent_frac = []

    fc3_latent_rws = []
    fc3_latent_mcc = []
    fc3_latent_recall = []
    fc3_latent_precision = []
    fc3_latent_frac = []

    fc4_latent_rws = []
    fc4_latent_mcc = []
    fc4_latent_recall = []
    fc4_latent_precision = []
    fc4_latent_frac = []

    for i, pth in enumerate(test_pth):
        test_ds = ImageFolderWithPaths(pth, transform=_test_transform)

        fc1_train_ds = ImageFolder(fc1_path, transform=_transform)
        fc2_train_ds = ImageFolder(fc2_path, transform=_transform)
        fc3_train_ds = ImageFolder(fc3_path, transform=_transform)
        fc4_train_ds = ImageFolder(fc4_path, transform=_transform)

        test_ds = ImageFolderWithPaths(pth, transform=_test_transform)
        fc1_sampler = balance_classes(fc1_train_ds)
        fc2_sampler = balance_classes(fc2_train_ds)
        fc3_sampler = balance_classes(fc3_train_ds)
        fc4_sampler = balance_classes(fc4_train_ds)

        fc1_train_loader = torch.utils.data.DataLoader(
            fc1_train_ds,
            sampler=fc1_sampler,
            batch_size=batch_size_train,
            drop_last=True,
        )
        fc2_train_loader = torch.utils.data.DataLoader(
            fc2_train_ds,
            sampler=fc2_sampler,
            batch_size=batch_size_train,
            drop_last=True,
        )
        fc3_train_loader = torch.utils.data.DataLoader(
            fc3_train_ds,
            sampler=fc3_sampler,
            batch_size=batch_size_train,
            drop_last=True,
        )
        fc4_train_loader = torch.utils.data.DataLoader(
            fc4_train_ds,
            sampler=fc4_sampler,
            batch_size=batch_size_train,
            drop_last=True,
        )

        training_loop(
            n_epoch,
            optim,
            Net,
            fc1_train_loader,
            learning_rate,
            momentum,
            saved_model_path,
            train,
            device,
        )

        network_saved = load_model(Net, saved_model_path)

        test_paths, labels_np, scores_decision = results_from_latent(
            network_saved,
            fc1_train_ds,
            test_ds,
            _test_transform,
            anomaly_label,
            device,
            layer=network_saved.fc1,
        )
        selected_anomalies = n_most_anomalous_images(scores_decision, test_paths, num_to_display)
        add_anomalies_to_training(selected_anomalies, fc1_path)

        y_true, y_pred = get_true_and_pred_labels(labels_np, scores_decision)

        _fc1_latent_rws = rws_score(labels_np, scores_decision)
        _fc1_latent_mcc = matthews_corrcoef(y_true, y_pred)
        _fc1_latent_recall = recall_score(y_true, y_pred)
        _fc1_latent_precision = precision_score(y_true, y_pred)
        _fc1_latent_frac = fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label)

        fc1_latent_mcc.append(_fc1_latent_mcc)
        fc1_latent_rws.append(_fc1_latent_rws)
        fc1_latent_recall.append(_fc1_latent_recall)
        fc1_latent_precision.append(_fc1_latent_precision)
        fc1_latent_frac.append(_fc1_latent_frac)
        dict_to_df(get_file_len(fc1_path))

        training_loop(
            n_epoch,
            optim,
            Net,
            fc2_train_loader,
            learning_rate,
            momentum,
            saved_model_path,
            train,
            device,
        )
        network_saved = load_model(Net, saved_model_path)
        test_paths, labels_np, scores_decision = results_from_latent(
            network_saved,
            fc2_train_ds,
            test_ds,
            _test_transform,
            anomaly_label,
            device,
            layer=network_saved.fc2,
        )
        selected_anomalies = n_most_anomalous_images(scores_decision, test_paths, num_to_display)
        add_anomalies_to_training(selected_anomalies, fc2_path)

        y_true, y_pred = get_true_and_pred_labels(labels_np, scores_decision)

        _fc2_latent_rws = rws_score(labels_np, scores_decision)
        _fc2_latent_mcc = matthews_corrcoef(y_true, y_pred)
        _fc2_latent_recall = recall_score(y_true, y_pred)
        _fc2_latent_precision = precision_score(y_true, y_pred)
        _fc2_latent_frac = fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label)

        fc2_latent_mcc.append(_fc2_latent_mcc)
        fc2_latent_rws.append(_fc2_latent_rws)
        fc2_latent_recall.append(_fc2_latent_recall)
        fc2_latent_precision.append(_fc2_latent_precision)
        fc2_latent_frac.append(_fc2_latent_frac)
        dict_to_df(get_file_len(fc2_path))

        training_loop(
            n_epoch,
            optim,
            Net,
            fc3_train_loader,
            learning_rate,
            momentum,
            saved_model_path,
            train,
            device,
        )
        network_saved = load_model(Net, saved_model_path)
        test_paths, labels_np, scores_decision = results_from_latent(
            network_saved,
            fc3_train_ds,
            test_ds,
            _test_transform,
            anomaly_label,
            device,
            layer=network_saved.fc3,
        )
        selected_anomalies = n_most_anomalous_images(scores_decision, test_paths, num_to_display)
        add_anomalies_to_training(selected_anomalies, fc3_path)

        y_true, y_pred = get_true_and_pred_labels(labels_np, scores_decision)

        _fc3_latent_rws = rws_score(labels_np, scores_decision)
        _fc3_latent_mcc = matthews_corrcoef(y_true, y_pred)
        _fc3_latent_recall = recall_score(y_true, y_pred)
        _fc3_latent_precision = precision_score(y_true, y_pred)
        _fc3_latent_frac = fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label)

        fc3_latent_mcc.append(_fc3_latent_mcc)
        fc3_latent_rws.append(_fc3_latent_rws)
        fc3_latent_recall.append(_fc3_latent_recall)
        fc3_latent_precision.append(_fc3_latent_precision)
        fc3_latent_frac.append(_fc3_latent_frac)
        dict_to_df(get_file_len(fc3_path))

        training_loop(
            n_epoch,
            optim,
            Net,
            fc4_train_loader,
            learning_rate,
            momentum,
            saved_model_path,
            train,
            device,
        )
        network_saved = load_model(Net, saved_model_path)
        test_paths, labels_np, scores_decision = results_from_latent(
            network_saved,
            fc4_train_ds,
            test_ds,
            _test_transform,
            anomaly_label,
            device,
            layer=network_saved.fc4,
        )
        selected_anomalies = n_most_anomalous_images(scores_decision, test_paths, num_to_display)
        add_anomalies_to_training(selected_anomalies, fc4_path)

        y_true, y_pred = get_true_and_pred_labels(labels_np, scores_decision)

        _fc4_latent_rws = rws_score(labels_np, scores_decision)
        _fc4_latent_mcc = matthews_corrcoef(y_true, y_pred)
        _fc4_latent_recall = recall_score(y_true, y_pred)
        _fc4_latent_precision = precision_score(y_true, y_pred)
        _fc4_latent_frac = fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label)

        fc4_latent_mcc.append(_fc4_latent_mcc)
        fc4_latent_rws.append(_fc4_latent_rws)
        fc4_latent_recall.append(_fc4_latent_recall)
        fc4_latent_precision.append(_fc4_latent_precision)
        fc4_latent_frac.append(_fc4_latent_frac)
        dict_to_df(get_file_len(fc4_path))

    return (
        fc1_latent_rws,
        fc1_latent_mcc,
        fc1_latent_recall,
        fc1_latent_precision,
        fc1_latent_frac,
        fc2_latent_rws,
        fc2_latent_mcc,
        fc2_latent_recall,
        fc2_latent_precision,
        fc2_latent_frac,
        fc3_latent_rws,
        fc3_latent_mcc,
        fc3_latent_recall,
        fc3_latent_precision,
        fc3_latent_frac,
        fc4_latent_rws,
        fc4_latent_mcc,
        fc4_latent_recall,
        fc4_latent_precision,
        fc4_latent_frac,
    )


def compute_iforest_rws(labels, scores):
    isof_rws_dec = rws_score(labels, scores)
    return isof_rws_dec


def get_true_and_pred_labels(labels_np, scores_decision):
    n_anomalies = np.sum(labels_np)

    threshold = np.sort(scores_decision)[-n_anomalies - 1]

    y_pred = scores_decision > threshold
    y_true = labels_np.astype(int)

    return y_true, y_pred


def compute_iforest_mcc(labels_np, scores):

    scores = scores.max() - scores
    n_anomalies = np.sum(labels_np)
    threshold = np.sort(scores)[-n_anomalies - 1]
    y_pred = scores > threshold

    return matthews_corrcoef(labels_np, y_pred)


def compute_ahunt_rws(actual_label, predict_proba, anomaly_label):
    """Computes rws score per round
    Parameters
    ----------
    actual_label: List
        Labels of the samples
    predict_proba: List
        Predicted probabilities of model
    """
    labels = [True if x == anomaly_label else False for x in actual_label]
    _rws_score = rws_score(labels, predict_proba)
    return _rws_score


def compute_ahunt_mcc(actual_label, pred_proba, anomaly_label):
    pred_proba = np.array(pred_proba)
    labels = [True if x == anomaly_label else False for x in actual_label]
    scores = pred_proba.max() - pred_proba
    n_anomalies = np.sum(np.array(labels))
    threshold = np.sort(scores)[-n_anomalies - 1]
    y_pred = scores > threshold

    return matthews_corrcoef(labels, y_pred)


def get_anomaly_index(test_loader, anomaly_label):
    class_to_idx = test_loader.dataset.class_to_idx
    for x, y in class_to_idx.items():
        if x == anomaly_label:
            return y
    return "Anomaly class not found"


def compute_latent_rws(labels, scores):
    scores = scores.max() - scores
    rws_dec = rws_score(labels, scores)
    return rws_dec


def compute_latent_mcc(labels_np, scores):
    scores = scores.max() - scores
    n_anomalies = np.sum(labels_np)
    threshold = np.sort(scores)[-n_anomalies - 1]
    y_pred = scores > threshold

    return matthews_corrcoef(labels_np, y_pred)


def get_iforest(i, testing_path, _transforms, train_iforest_path, anomaly_label, iforest_use_all):

    print(f"{color.PURPLE}Iforest for Night {i+1}{color.END}")

    train_ds, test_ds = (
        ImageFolder(train_iforest_path, transform=_transforms),
        ImageFolder(testing_path, transform=_transforms),
    )

    (
        labels_np,
        scores_decision,
        test_paths,
    ) = results_from_iforest(train_ds, test_ds, iforest_use_all, anomaly_label)

    num_to_display = np.sum(labels_np)

    selected_anomalies = n_most_anomalous_images(scores_decision, test_paths, num_to_display)

    add_anomalies_to_training(selected_anomalies, train_iforest_path)

    y_true, y_pred = get_true_and_pred_labels(labels_np, scores_decision)

    _iso_rws = rws_score(labels_np, scores_decision)
    _iso_mcc = matthews_corrcoef(y_true, y_pred)
    _iso_recall = recall_score(y_true, y_pred)
    _iso_precision = precision_score(y_true, y_pred)
    _iso_frac = fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label)

    dict_to_df(get_file_len(train_iforest_path))

    return _iso_rws, _iso_mcc, _iso_recall, _iso_precision, _iso_frac


def get_latent(
    i,
    testing_path,
    _transforms,
    train,
    n_epoch,
    optim,
    Net,
    learning_rate,
    momentum,
    saved_model_path,
    train_latent_path,
    batch_size_train,
    batch_size_test,
    anomaly_label,
    loss,
    query_strategy,
    train_once,
    device,
):
    print(f"{color.PURPLE}Latent Space for Night {i +1}{color.END}")

    train_ds = ImageFolder(train_latent_path, transform=_transforms)
    test_ds = ImageFolderWithPaths(testing_path, transform=_transforms)
    sampler = balance_classes(train_ds)

    train_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=batch_size_train, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=True)

    if train_once:
        if i == 0:
            _, network_saved = training_loop(
                n_epoch,
                optim,
                Net,
                train_loader,
                learning_rate,
                momentum,
                saved_model_path,
                train,
                loss,
                device,
            )
        else:
            Net.load_state_dict(torch.load(saved_model_path))
            network_saved = Net
    else:
        _, network_saved = training_loop(
            n_epoch,
            optim,
            Net,
            train_loader,
            learning_rate,
            momentum,
            saved_model_path,
            train,
            loss,
            device,
        )

    if network_saved._get_name() == "ResNet":
        layer = network_saved.avgpool
    else:
        layer = network_saved.fc1

    all_test_paths, labels_np, scores_decision, test_labels = results_from_latent(
        network_saved, train_ds, test_ds, _transforms, anomaly_label, device, layer
    )

    num_to_display = np.sum(labels_np)

    y_true, y_pred, selected_anomalies = output_query_strategy(
        query_strategy,
        scores_decision,
        all_test_paths,
        num_to_display,
        labels_np,
    )

    add_anomalies_to_training(selected_anomalies, train_latent_path)

    y_true, y_pred = get_true_and_pred_labels(labels_np, scores_decision)

    _latent_rws = rws_score(labels_np, scores_decision)
    _latent_mcc = matthews_corrcoef(y_true, y_pred)
    _latent_recall = recall_score(y_true, y_pred)
    _latent_precision = precision_score(y_true, y_pred)
    _latent_frac = fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label)

    dict_to_df(get_file_len(train_latent_path))

    return _latent_rws, _latent_mcc, _latent_recall, _latent_precision, _latent_frac


def get_ahunt(
    i,
    testing_path,
    _transforms,
    train,
    n_epoch,
    optim,
    Net,
    learning_rate,
    momentum,
    saved_model_path,
    train_ahunt_path,
    batch_size_train,
    batch_size_test,
    anomaly_label,
    initial_training_config,
    loss,
    query_strategy,
    train_once,
    device,
):
    print(f"{color.PURPLE}Ahunt for Night {i +1}{color.END}")

    train_ds = ImageFolder(train_ahunt_path, transform=_transforms)
    sampler = balance_classes(train_ds)

    train_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, batch_size=batch_size_train, drop_last=True)

    if train_once:
        if i == 0:
            _, network_saved = training_loop(
                n_epoch,
                optim,
                Net,
                train_loader,
                learning_rate,
                momentum,
                saved_model_path,
                train,
                loss,
                device,
            )
        else:
            Net.load_state_dict(torch.load(saved_model_path))
            network_saved = Net
    else:
        _, network_saved = training_loop(
            n_epoch,
            optim,
            Net,
            train_loader,
            learning_rate,
            momentum,
            saved_model_path,
            train,
            loss,
            device,
        )

    test_ds = ImageFolderWithPaths(testing_path, transform=_transforms)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=True)
    normal_classes = get_normal_classes(initial_training_config, anomaly_label)
    indices = get_class_indices(train_ds, normal_classes)
    index_separator = get_midpoint(indices)

    anomaly_idx = get_anomaly_idx(test_loader, anomaly_label)

    (
        all_test_paths,
        labels_np,
        scores_decision,
        prob_difference,
        actual_labels,
        predicted_labels,
        all_probs,
        all_outputs,
    ) = test(network_saved, test_loader, anomaly_label, anomaly_idx, index_separator, device)

    num_to_display = np.sum(labels_np)

    y_true, y_pred, selected_anomalies = output_query_strategy(
        query_strategy,
        scores_decision,
        all_test_paths,
        num_to_display,
        labels_np,
    )

    add_anomalies_to_training(selected_anomalies, train_ahunt_path)

    _ahunt_rws = rws_score(labels_np, scores_decision)
    _ahunt_mcc = matthews_corrcoef(y_true, y_pred)
    _ahunt_recall = recall_score(y_true, y_pred)
    _ahunt_precision = precision_score(y_true, y_pred)
    _ahunt_frac = fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label)

    # print(f"Fraction of Anomalies==={_ahunt_frac} and MCC Score === {_ahunt_mcc}")

    dict_to_df(get_file_len(train_ahunt_path))

    return _ahunt_rws, _ahunt_mcc, _ahunt_recall, _ahunt_precision, _ahunt_frac


def get_combined_output(
    test_pth,
    _transforms,
    iforest_use_all,
    train_iforest_path,
    train_latent_path,
    train_ahunt_path,
    batch_size_train,
    batch_size_test,
    n_epoch,
    optim,
    Net,
    learning_rate,
    momentum,
    train,
    saved_model_path,
    anomaly_label,
    collective_test_path,
    use_cummulative_test,
    initial_training_config,
    loss,
    query_strategy,
    train_once,
    device,
):
    iso_rws = []
    iso_mcc = []
    iso_recall = []
    iso_precision = []
    iso_frac = []

    latent_rws = []
    latent_mcc = []
    latent_recall = []
    latent_precision = []
    latent_frac = []

    ahunt_rws = []
    ahunt_mcc = []
    ahunt_recall = []
    ahunt_precision = []
    ahunt_frac = []

    for i, pth in enumerate(test_pth):

        if use_cummulative_test:
            testing_path = collective_test_path
            move_files(pth, testing_path)

        else:
            testing_path = pth

        _iso_rws, _iso_mcc, _iso_recall, _iso_precision, _iso_frac = get_iforest(
            i, testing_path, _transforms, train_iforest_path, anomaly_label, iforest_use_all
        )
        iso_rws.append(_iso_rws)
        iso_mcc.append(_iso_mcc)
        iso_recall.append(_iso_recall)
        iso_precision.append(_iso_precision)
        iso_frac.append(_iso_frac)

        _latent_rws, _latent_mcc, _latent_recall, _latent_precision, _latent_frac = get_latent(
            i,
            testing_path,
            _transforms,
            train,
            n_epoch,
            optim,
            Net,
            learning_rate,
            momentum,
            saved_model_path,
            train_latent_path,
            batch_size_train,
            batch_size_test,
            anomaly_label,
            loss,
            query_strategy,
            train_once,
            device,
        )
        latent_mcc.append(_latent_mcc)
        latent_rws.append(_latent_rws)
        latent_recall.append(_latent_recall)
        latent_precision.append(_latent_precision)
        latent_frac.append(_latent_frac)

        _ahunt_rws, _ahunt_mcc, _ahunt_recall, _ahunt_precision, _ahunt_frac = get_ahunt(
            i,
            testing_path,
            _transforms,
            train,
            n_epoch,
            optim,
            Net,
            learning_rate,
            momentum,
            saved_model_path,
            train_ahunt_path,
            batch_size_train,
            batch_size_test,
            anomaly_label,
            initial_training_config,
            loss,
            query_strategy,
            train_once,
            device,
        )

        ahunt_rws.append(_ahunt_rws)
        ahunt_mcc.append(_ahunt_mcc)
        ahunt_recall.append(_ahunt_recall)
        ahunt_precision.append(_ahunt_precision)
        ahunt_frac.append(_ahunt_frac)

    return (
        iso_rws,
        iso_mcc,
        iso_recall,
        iso_precision,
        iso_frac,
        latent_rws,
        latent_mcc,
        latent_recall,
        latent_precision,
        latent_frac,
        ahunt_rws,
        ahunt_mcc,
        ahunt_recall,
        ahunt_precision,
        ahunt_frac,
    )


def output_query_strategy(
    query_strategy,
    scores_decision,
    all_test_paths,
    num_to_display,
    labels_np,
):

    if query_strategy == "most_anomalous":
        selected_anomalies = n_most_anomalous_images(scores_decision, all_test_paths, num_to_display)
        y_true, y_pred = get_true_and_pred_labels(labels_np, scores_decision)
    elif query_strategy == "most_confused":
        confused_scores = m_confused_score(scores_decision)
        selected_anomalies = n_most_confused(confused_scores, all_test_paths, num_to_display)
        y_true, y_pred = get_true_and_pred_labels(labels_np, confused_scores)
    elif query_strategy == "from_lowest":
        selected_anomalies = n_from_lowest(scores_decision, all_test_paths, num_to_display)
        y_true, y_pred = get_true_and_pred_labels(labels_np, scores_decision)

    return y_true, y_pred, selected_anomalies


def evaluate_model(model, test_loader):
    actuals, predictions = torch.tensor([]), torch.tensor([])
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, (data, target, _) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            preds = output.argmax(dim=1).to("cpu")
            predictions = torch.cat((predictions, preds), dim=0)
            actuals = torch.cat((actuals, target.to("cpu")), dim=0)
    return actuals, predictions


def get_normal_classes(initial_data_config, anomaly_label):
    normal_classes = list(initial_data_config.keys() - anomaly_label)
    return normal_classes


def get_class_indices(train_ds, normal_classes):
    return (
        train_ds.class_to_idx[normal_classes[0]],
        train_ds.class_to_idx[normal_classes[1]],
    )


def get_midpoint(indices):
    return max(indices) + 1


def results_from_iforest_static(test_ds, anomaly_label):
    from pathlib import Path

    #     train_np_imgs, train_labels, _ = get_np_imgs(train_ds)
    test_np_imgs, test_labels, test_paths = get_np_imgs(test_ds)

    iforest = IsolationForest()
    iforest.fit(test_np_imgs)

    scores_decision = iforest.decision_function(test_np_imgs)

    scores_decision = scores_decision.max() - scores_decision

    labels_np = np.array([True if Path(x).parent.name == anomaly_label else False for x in test_paths])
    return labels_np, scores_decision, test_paths


def basic_reset_medmnist(_std, initial_training_config, initial_data_config, rounds, anomaly_label, add_noise):
    var = _std ** 2

    all_data_config = get_basic_config(rounds, initial_data_config, anomaly_label)

    src_original = "data/medmnist/medical-mnist"
    dest_pool = "data/medmnist/pool"
    medmnist_test = "data/medmnist/testing"
    src = "data/medmnist/pool"
    medmnist_main = "data/medmnist"

    zero_folders(
        "data/medmnist/pool",
        "data/medmnist/testing",
        "data/medmnist/training",
        "data/medmnist/testing_main",
        src=src_original,
        dest=dest_pool,
    )

    training_path = "data/medmnist/training/"
    initial_training(
        "data/medmnist/pool",
        training_path,
        initial_training_config,
        var=var,
        add_noise=add_noise,
    )

    create_ds_all_config(medmnist_test, src, all_data_config, var=var, add_noise=add_noise)
    create_test_folders(initial_data_config, medmnist_main)


def basic_reset_cifar(_std, initial_training_config, initial_data_config, rounds, anomaly_label, add_noise):
    var = _std ** 2
    all_data_config = get_basic_config(rounds, initial_data_config, anomaly_label)

    pool_src = "data/cifar/cifar-10/train"
    pool_path = "data/cifar/pool"
    training_path = "data/cifar/training/"
    test_path = "data/cifar/testing/"
    testing_main = "data/cifar/testing_main/"
    cifar_main = "data/cifar"

    zero_folders(
        pool_path,
        test_path,
        training_path,
        testing_main,
        src=pool_src,
        dest=pool_path,
    )
    labels = ["horse", "dog"]
    prefix_files_with_labels(pool_path)
    combine_classes(pool_path, labels, anomaly_label, 0.5)

    initial_training(
        pool_path,
        training_path,
        initial_training_config,
        var=var,
        add_noise=add_noise,
    )
    create_ds_all_config(test_path, pool_path, all_data_config, var=var, add_noise=add_noise)

    create_test_folders(initial_data_config, cifar_main)


def basic_reset_cifar_one(_std, initial_training_config, initial_data_config, rounds, anomaly_label, add_noise):
    var = _std ** 2
    all_data_config = get_basic_config(rounds, initial_data_config, anomaly_label)

    pool_src = "data/cifar/cifar-10/train"
    pool_path = "data/cifar/pool"
    training_path = "data/cifar/training/"
    test_path = "data/cifar/testing/"
    testing_main = "data/cifar/testing_main/"
    cifar_main = "data/cifar"

    zero_folders(
        pool_path,
        test_path,
        training_path,
        testing_main,
        src=pool_src,
        dest=pool_path,
    )

    initial_training(
        pool_path,
        training_path,
        initial_training_config,
        var=var,
        add_noise=add_noise,
    )
    create_ds_all_config(test_path, pool_path, all_data_config, var=var, add_noise=add_noise)

    create_test_folders(initial_data_config, cifar_main)


def basic_reset_cifar_infinite(_std, initial_training_config, initial_data_config, rounds, anomaly_label, add_noise):
    var = _std ** 2
    all_data_config = get_basic_config(rounds, initial_data_config, anomaly_label)

    pool_src = "data/cifar/cifar-10/train"
    pool_path = "data/cifar/pool"
    training_path = "data/cifar/training/"
    test_path = "data/cifar/testing/"
    testing_main = "data/cifar/testing_main/"
    cifar_main = "data/cifar"

    zero_folders(
        pool_path,
        test_path,
        training_path,
        testing_main,
        src=pool_src,
        dest=pool_path,
    )
    labels = [
        "automobile",
        "ship",
        "dog",
        "cat",
        "horse",
        "frog",
        "deer",
        "truck",
    ]
    prefix_files_with_labels(pool_path)
    combine_classes(pool_path, labels, anomaly_label, 0.1)

    initial_training(
        pool_path,
        training_path,
        initial_training_config,
        var=var,
        add_noise=add_noise,
    )
    create_ds_all_config(test_path, pool_path, all_data_config, var=var, add_noise=add_noise)

    create_test_folders(initial_data_config, cifar_main)


def ifrorest_comparison(
    test_pth,
    train_iforest_path,
    num_to_display,
    _transform,
    anomaly_label,
):
    iso_static_rws = []
    iso_static_mcc = []
    iso_static_recall = []
    iso_static_precision = []
    iso_static_frac = []

    iso_learning_rws = []
    iso_learning_mcc = []
    iso_learning_recall = []
    iso_learning_precision = []
    iso_learning_frac = []

    for i, pth in enumerate(test_pth):

        train_ds, test_ds = (
            ImageFolder(train_iforest_path, transform=_transform),
            ImageFolder(pth, transform=_transform),
        )

        (
            labels_np,
            scores_decision,
            test_paths,
        ) = results_from_iforest_static(test_ds, anomaly_label)

        selected_anomalies = n_most_anomalous_images(scores_decision, test_paths, num_to_display)

        (
            _rws_static,
            _mcc_static,
            _recall_static,
            _precision_static,
            _frac_static,
        ) = get_metrics(labels_np, scores_decision, selected_anomalies, anomaly_label)

        iso_static_rws.append(_rws_static)
        iso_static_mcc.append(_mcc_static)
        iso_static_recall.append(_recall_static)
        iso_static_precision.append(_precision_static)
        iso_static_frac.append(_frac_static)

        print(f"IForest learning Round {i +1}")
        train_ds, test_ds = (
            ImageFolder(train_iforest_path, transform=_transform),
            ImageFolder(pth, transform=_transform),
        )

        (
            labels_np,
            scores_decision,
            test_paths,
        ) = results_from_iforest(train_ds, test_ds, anomaly_label)

        selected_anomalies = n_most_anomalous_images(scores_decision, test_paths, num_to_display)
        add_anomalies_to_training(selected_anomalies, train_iforest_path)

        (
            _rws_learning,
            _mcc_learning,
            _recall_learning,
            _precision_learning,
            _frac_learning,
        ) = get_metrics(labels_np, scores_decision, selected_anomalies, anomaly_label)

        iso_learning_rws.append(_rws_learning)
        iso_learning_mcc.append(_mcc_learning)
        iso_learning_recall.append(_recall_learning)
        iso_learning_precision.append(_precision_learning)
        iso_learning_frac.append(_frac_learning)

        dict_to_df(get_file_len(train_iforest_path))
    return (
        iso_static_rws,
        iso_static_mcc,
        iso_static_recall,
        iso_static_precision,
        iso_static_frac,
        iso_learning_rws,
        iso_learning_mcc,
        iso_learning_recall,
        iso_learning_precision,
        iso_learning_frac,
    )


def get_isolation_forest(
    test_pth,
    train_iforest_path,
    num_to_display,
    _transform,
    anomaly_label,
):
    print("****************Getting Values for isolation forests**********")
    iso_rws = []
    iso_mcc = []
    iso_recall = []
    iso_precision = []
    iso_frac = []

    for i, pth in enumerate(test_pth):

        train_ds, test_ds = (
            ImageFolder(train_iforest_path, transform=_transform),
            ImageFolder(pth, transform=_transform),
        )

        (
            labels_np,
            scores_decision,
            test_paths,
        ) = results_from_iforest(train_ds, test_ds, anomaly_label)

        selected_anomalies = n_most_anomalous_images(scores_decision, test_paths, num_to_display)
        add_anomalies_to_training(selected_anomalies, train_iforest_path)

        y_true, y_pred = get_true_and_pred_labels(labels_np, scores_decision)

        _iso_rws = rws_score(labels_np, scores_decision)
        _iso_mcc = matthews_corrcoef(y_true, y_pred)
        _iso_recall = recall_score(y_true, y_pred)
        _iso_precision = precision_score(y_true, y_pred)
        _iso_frac = fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label)

        iso_rws.append(_iso_rws)
        iso_mcc.append(_iso_mcc)
        iso_recall.append(_iso_recall)
        iso_precision.append(_iso_precision)
        iso_frac.append(_iso_frac)

        dict_to_df(get_file_len(train_iforest_path))

    return iso_rws, iso_mcc, iso_recall, iso_precision, iso_frac


def add_noise(pth, var):
    import skimage
    import skimage.io

    origin = skimage.io.imread(pth)
    noisy = skimage.util.random_noise(origin, mode="gaussian", var=var)

    return noisy


def np_to_img(pth_to_save, np_array):
    return plt.imsave(pth_to_save, np_array, cmap="Greys")


def cp_and_add_noise(src, dest, file_names, var):
    for fl_name in file_names:
        img_src_path = os.path.join(src, fl_name)
        noisy_img = add_noise(img_src_path, var)
        dest_img = f"{dest}/{fl_name}"
        np_to_img(dest_img, noisy_img)

        ### remove image from source - pick without replacement
        ### pathlib.Path(os.path.join(src, fl_name)).unlink()


def latent_most_anomalous(_test_paths, scores_decision, num_to_display, anomaly_label):
    accurate_score_decision, accurate_anomaly_path = [], []

    _anomalous_probaba, _anomalous_paths = zip(*sorted(zip(scores_decision, _test_paths), reverse=False))

    selected_test_path, selected_score_decision = _anomalous_paths[:num_to_display], _anomalous_probaba[:num_to_display]

    for path, decision in zip(selected_test_path, selected_score_decision):
        _label = Path(path).parent.name
        if _label == anomaly_label:
            accurate_anomaly_path.append(path)
            accurate_score_decision.append(decision)

    return accurate_anomaly_path, accurate_score_decision


def results_from_latent(model, train_ds, test_ds, transform, anomaly_label, device, layer):

    name = "latent"

    test_activations, test_labels, test_paths = get_latent_activations(model, name, layer, transform, test_ds, device)
    train_activations, train_labels, _ = get_latent_activations(model, name, layer, transform, train_ds, device)
    train_activations = [x.tolist() for x in train_activations]
    test_activations = [x.tolist() for x in test_activations]

    iforest = IsolationForest()
    iforest.fit(np.array(train_activations))

    scores_decision = iforest.decision_function(test_activations)  # the lower, the more anomalous

    scores_decision = scores_decision.max() - scores_decision

    labels_np = np.array([True if Path(x).parent.name == anomaly_label else False for x in test_paths])

    return test_paths, labels_np, scores_decision, test_labels


def results_from_latent_trial(model, train_ds, test_ds, anomaly_label, device, layer):

    # name = "latent"
    batch_size_train = batch_size_test = 64

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_train)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=False)

    train_activations, test_activations, test_paths, labels_np = get_all_activations(
        model, train_loader, test_loader, device, layer, anomaly_label
    )

    iforest = IsolationForest()
    iforest.fit(train_activations)

    scores_decision = iforest.decision_function(test_activations)  # the lower, the more anomalous

    scores_decision = scores_decision.max() - scores_decision

    return test_paths, labels_np, scores_decision


def get_all_activations(model, train_loader, test_loader, device, layer, anomaly_label):
    model.eval()

    train_activations, _, _ = get_latent_activation(model, train_loader, device, layer, loader_type="train")
    test_activations, test_labels, test_paths = get_latent_activation(
        model, test_loader, device, layer, loader_type="test"
    )
    anomaly_idx = get_anomaly_idx(test_loader, anomaly_label)
    labels_np = np.array([True if x == anomaly_idx else False for x in test_labels])
    return train_activations, test_activations, test_paths, labels_np


def get_latent_activation(model, data_loader, device, layer, loader_type):
    latent_list = []
    target_list = []
    test_path_list = []

    model.eval()

    def get_activation(name):
        # the hook signature
        model.eval()

        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    activation = {}
    h1 = layer.register_forward_hook(get_activation("fc1"))

    with torch.no_grad():
        if loader_type == "train":
            for data, target in data_loader:
                # forward pass -- getting the outputs
                data = data.to(device)
                out = model(data)  # .to(device)

                # collect the activations in the correct list
                batch_activations = torch.squeeze(activation["fc1"])

                latent_list.extend(batch_activations.cpu().detach().numpy())
                target_list.extend(target.tolist())

        else:
            for data, target, path in data_loader:
                # forward pass -- getting the outputs
                data = data.to(device)
                out = model(data)  # .to(device)

                # collect the activations in the correct list
                batch_activations = torch.squeeze(activation["fc1"])

                latent_list.extend(batch_activations.cpu().detach().numpy())
                target_list.extend(target.tolist())
                test_path_list.extend(path)
    h1.remove()

    return latent_list, target_list, test_path_list


def get_latent_activations(model, name, layer, transform, ds, device):
    """Returns Latent Activations"""

    model = model.to(device_cpu)
    activations = []
    img_paths = [x[0] for x in ds.samples]
    img_labels = [x[1] for x in ds.samples]

    for path in img_paths:
        activation = get_layer_activation(model, name, layer, transform, path, device)
        activations.append(activation)
    return activations, img_labels, img_paths


def get_layer_activation(model, name, layer, transform, path, device):
    """Returns Layer Activation"""
    model.eval()
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    h1 = layer.register_forward_hook(get_activation(name))
    img = tensorify_img(path, transform)

    output = model(img[None, ...].float()).to(device)

    h1.remove()

    return activation[name].reshape(-1)


def get_anomaly_idx(dataloader, anomaly_label):
    return dataloader.dataset.class_to_idx[anomaly_label]


def plot_loss_single(training_loss):
    _num_of_epochs = len(training_loss) + 1
    epoch_nums = [x for x in range(1, _num_of_epochs)]
    plt.figure(figsize=(8, 8))
    plt.plot(epoch_nums, training_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    #     plt.legend(
    #         [
    #             "training",
    #         ],
    #         loc="upper right",
    #     )
    plt.show()


def tensorify_img(path, transform):
    img = Image.open(path)

    if is_grayscale(path, transform):
        img = Image.open(path).convert("RGB")

    img_tensor = transform(img)
    return img_tensor


def is_grayscale(path, transform):
    return transform(Image.open(path)).shape[0] == 1


def cp_detected_anomalies(anomalies_detected, dest_anomaly_folder):
    for i in anomalies_detected:
        shutil.copy2(i, dest_anomaly_folder)


def rws_score(is_outlier, outlier_score, n_o=None):
    outliers = np.array(is_outlier)
    if n_o is None:
        n_o = int(np.sum(outliers))
    b_s = np.arange(n_o) + 1
    o_ind = np.argsort(outlier_score)[-n_o:]
    if b_s.size == 0:
        return 0

    return 1.0 * np.sum(b_s * outliers[o_ind].reshape(-1)) / np.sum(b_s)


def create_latent_training(latent_training_path):

    folders_to_create = ["fc1", "fc2", "fc3", "fc4"]

    paths = [f"{latent_training_path}/{folder}" for folder in folders_to_create]

    for path in paths:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return paths


def initial_latent_training(src_pool, latent_training_path, init_config):
    # Creates folders for Iforest,Latent and Ahunt
    _fc1, _fc2, _fc3, _fc4 = create_latent_training(latent_training_path)

    for folder, size in init_config.items():

        label_src = f"{src_pool}/{folder}"

        fc1 = f"{_fc1}/{folder}"
        fc2 = f"{_fc2}/{folder}"
        fc3 = f"{_fc3}/{folder}"
        fc4 = f"{_fc4}/{folder}"

        pathlib.Path(fc1).mkdir(parents=True, exist_ok=True)
        pathlib.Path(fc2).mkdir(parents=True, exist_ok=True)
        pathlib.Path(fc3).mkdir(parents=True, exist_ok=True)
        pathlib.Path(fc4).mkdir(parents=True, exist_ok=True)

        _, fl_names = get_random_fls(label_src, num_files=size)

        cp_files(label_src, fc1, fl_names)
        cp_files(label_src, fc2, fl_names)
        cp_files(label_src, fc3, fl_names)
        cp_files(label_src, fc4, fl_names)

        rm_files(label_src, fl_names)

    print("fc1")
    dict_to_df(get_file_len(f"{_fc1}/"))
    print("fc2")
    dict_to_df(get_file_len(f"{_fc2}/"))
    print("fc3")
    dict_to_df(get_file_len(f"{_fc3}/"))
    print("fc4")
    dict_to_df(get_file_len(f"{_fc4}/"))


def _get_individual_training_paths(training_path):
    iforest_path = f"{training_path}iforest/"
    latent_path = f"{training_path}latent/"
    ahunt_path = f"{training_path}ahunt/"

    return iforest_path, latent_path, ahunt_path


def is_test_fl_unique(test_pth):
    import collections

    all_data = collections.defaultdict(list)
    label_names = [f.name for f in test_pth[0].iterdir() if f.is_dir()]

    for pth in test_pth:
        for label in label_names:
            pth_label = f"{pth}/{label}"
            fl_names = os.listdir(pth_label)
            all_data[label].extend(fl_names)

    for k, v in all_data.items():
        if len(v) == len(set(v)):
            print(f"Label {k} is unique")
        else:
            print(f"Label {k} isn't unique")


def move_files(pth, destination_path):

    path_to_labels = [f for f in pth.iterdir() if f.is_dir()]

    for label_path in path_to_labels:
        fl_names = os.listdir(label_path)
        label_name = label_path.name
        dest_path = f"{destination_path}/{label_name}"

        for fl_name in fl_names:
            shutil.move(os.path.join(label_path, fl_name), dest_path)


def create_test_folders(initial_test_config, test_main):
    rm_folders(f"{test_main}/testing_main")
    lst_folders = ["iforest", "latent", "ahunt"]
    labels = initial_test_config.keys()
    for i in lst_folders:
        fld = f"{test_main}/testing_main/{i}"
        pathlib.Path(fld).mkdir(parents=True, exist_ok=True)

        for label in labels:
            fld_label = f"{fld}/{label}"
            pathlib.Path(fld_label).mkdir(parents=True, exist_ok=True)


def create_dir_training(training_path):
    iforest_path, latent_path, ahunt_path = _get_individual_training_paths(training_path)
    dir_to_create = [iforest_path, latent_path, ahunt_path]
    for path in dir_to_create:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    return iforest_path, latent_path, ahunt_path


def cp_files(src, dest, fl_names):
    import shutil

    for fl in fl_names:
        fl_path = os.path.join(src, fl)
        shutil.copy2(fl_path, dest)


def rm_files(src, fl_names):
    for fl in fl_names:
        pathlib.Path(os.path.join(src, fl)).unlink()


def create_ds_all_config(path, src, all_data_config, var=None, add_noise=False, round_name="round"):

    for i, data_config in enumerate(all_data_config):
        round_folder = f"{path}/{round_name}_{i}"
        pathlib.Path(round_folder).mkdir(parents=True, exist_ok=True)
        for folder, data_size in data_config.items():  # create folder for labels
            dest = f"{round_folder}/{folder}"
            label_src = f"{src}/{folder}"

            pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
            if data_size != 0:
                _, file_names = get_random_fls(label_src, num_files=data_size)
                if folder == "anomaly":
                    file_names = get_combined_anomalies(label_src, num_files=data_size)
                if folder == "infinite_anomalies":
                    file_names = get_n_combined_classes(label_src, num_files=data_size)
                if add_noise:
                    cp_and_add_noise(label_src, dest, file_names, var=var)
                else:
                    pass
                rm_files(label_src, file_names)  # pick without replacement

        dict_to_df(get_file_len(round_folder))


def initial_training(src_pool, training_path, init_config, var, add_noise=False):
    # Creates folders for Iforest,Latent and Ahunt
    iforest_path, latent_path, ahunt_path = create_dir_training(training_path)

    for folder, size in init_config.items():

        label_src = f"{src_pool}/{folder}"

        iso_dest = f"{iforest_path}/{folder}"
        latent_dest = f"{latent_path}/{folder}"
        ahunt_dest = f"{ahunt_path}/{folder}"

        pathlib.Path(iso_dest).mkdir(parents=True, exist_ok=True)
        pathlib.Path(latent_dest).mkdir(parents=True, exist_ok=True)
        pathlib.Path(ahunt_dest).mkdir(parents=True, exist_ok=True)

        if size != 0:
            _, fl_names = get_random_fls(label_src, num_files=size)

            if folder == "anomaly":
                fl_names = get_combined_anomalies(label_src, num_files=size)
            elif folder == "infinite_anomalies":
                fl_names = get_n_combined_classes(label_src, num_files=size)

            if add_noise:
                cp_and_add_noise(label_src, iso_dest, fl_names, var=var)
                cp_and_add_noise(label_src, latent_dest, fl_names, var=var)
                cp_and_add_noise(label_src, ahunt_dest, fl_names, var=var)
            else:
                cp_files(label_src, iso_dest, fl_names)
                cp_files(label_src, latent_dest, fl_names)
                cp_files(label_src, ahunt_dest, fl_names)

            rm_files(label_src, fl_names)
    print("Iforest")
    dict_to_df(get_file_len(iforest_path))
    print("Latent")
    dict_to_df(get_file_len(latent_path))
    print("Ahunt")
    dict_to_df(get_file_len(ahunt_path))


def init_training(src, training_path, init_config):
    import shutil

    if Path(training_path).is_dir():
        shutil.rmtree(training_path)

    iforest_path, latent_path, ahunt_path = _get_individual_training_paths(training_path)

    pathlib.Path(training_path).mkdir(parents=True, exist_ok=True)  # create main folder
    pathlib.Path(latent_path).mkdir(parents=True, exist_ok=True)  # create main folder
    pathlib.Path(ahunt_path).mkdir(parents=True, exist_ok=True)  # create main folder

    pathlib.Path(iforest_path).mkdir(parents=True, exist_ok=True)  # copy for iforest
    for folder, size in init_config.items():
        dest = f"{iforest_path}/{folder}"
        label_src = f"{src}/{folder}"

        pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
        _, file_names = get_random_fls(label_src, num_files=size)
        mv_files(label_src, dest, file_names)

    dict_to_df(get_file_len(iforest_path))

    shutil.copytree(f"{iforest_path}", latent_path)
    dict_to_df(get_file_len(latent_path))
    shutil.copytree(f"{iforest_path}", ahunt_path)
    dict_to_df(get_file_len(ahunt_path))


def reinitialize_training(training_path, test_pth):

    iforest_path, latent_path, ahunt_path = _get_individual_training_paths(training_path)

    shutil.rmtree(training_path)

    pathlib.Path(training_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(iforest_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(latent_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(ahunt_path).mkdir(parents=True, exist_ok=True)

    shutil.copytree(f"{test_pth[0]}", iforest_path, dirs_exist_ok=True)
    shutil.copytree(f"{test_pth[0]}", latent_path, dirs_exist_ok=True)
    shutil.copytree(f"{test_pth[0]}", ahunt_path, dirs_exist_ok=True)


def results_from_iforest(train_ds, test_ds, iforest_use_all, anomaly_label):
    from pathlib import Path

    train_np_imgs, _, _ = get_np_imgs(train_ds)
    test_np_imgs, _, test_paths = get_np_imgs(test_ds)
    np_imgs_all, _, _ = get_np_imgs_all(train_ds, test_ds)

    iforest = IsolationForest()

    if iforest_use_all:
        iforest.fit(np_imgs_all)
    else:
        iforest.fit(train_np_imgs)

    scores_decision = iforest.decision_function(test_np_imgs)

    scores_decision = scores_decision.max() - scores_decision

    labels_np = np.array([True if Path(x).parent.name == anomaly_label else False for x in test_paths])
    return labels_np, scores_decision, test_paths


def most_confused_score(all_outputs):
    normal = all_outputs[:, 0] + all_outputs[:, 1]
    anom = all_outputs[:, 2].tolist()
    normal = normal.tolist()

    sub = []

    for a, b in zip(anom, normal):
        if a > b:
            _sub = a - b
        else:
            _sub = b - a
        sub.append(_sub)

    outputs = torch.tensor(sub)
    return outputs


def rm_checkpoint(src):
    file_to_rem = pathlib.Path(src)
    file_to_rem.unlink()


def n_most_anomalous_images(scores_decision, test_paths, num_to_display):

    _, _anomalous_paths = zip(*sorted(zip(scores_decision, test_paths), reverse=True))

    selected_anomalies = _anomalous_paths[:num_to_display]

    return selected_anomalies


def m_confused_score(scores_decision):

    z_clf = np.array(scores_decision)
    z_clf[z_clf < 0.5] = 0

    # if np.sum(z_clf) == 0:
    #     return np.abs(0.5 - np.array(scores_decision))
    # else:
    confused = 1 - np.abs(0.5 - z_clf)
    return confused


def n_most_confused(scores_decision, test_paths, num_to_display):
    # anomalous_indices = np.array(prob_difference).argsort()[:num_to_display]
    result = np.max(scores_decision) == np.min(scores_decision) == 0.5
    _, _anomalous_paths = zip(*sorted(zip(scores_decision.tolist(), test_paths), reverse=True))

    selected_anomalies = _anomalous_paths[:num_to_display]

    if result:  # used when no anomaly is detected in the zeroth round
        selected_anomalies = mnist_selected(_anomalous_paths, label_count={"5": 1, "1": 2, "0": 3})

    # anomalous_indices = np.argsort(z_clf)[:num_to_display]  # [::-1]
    # selected_anomalies = [test_paths[index] for index in anomalous_indices]
    return selected_anomalies


def n_from_lowest(scores_decision, test_paths, num_to_display):
    z_clf = np.array(scores_decision)
    z_clf[z_clf < 0.5] = 0
    scr_ano = 1 - 2 * np.abs(z_clf - 0.5)
    anomalous_indices = np.argsort(scr_ano)[::-1]
    anomalous_indices_selected = anomalous_indices[:num_to_display]
    selected_anomalies = [test_paths[index] for index in anomalous_indices_selected]
    return selected_anomalies


def mnist_selected(_anomalous_paths, label_count={"5": 1, "1": 2, "0": 3}):
    import random

    label_table = {}
    for i in _anomalous_paths:
        label = pathlib.Path(i).parent.name
        if label not in label_table:
            label_table[label] = []
        label_table[label].append(i)

    selected = []
    for label in label_table:
        selected.extend(random.sample(label_table[label], label_count[label]))

    return selected


def select_random_n_anomalies(test_paths, num_to_display):
    import random

    selected_paths = random.sample(test_paths, num_to_display)

    return selected_paths


def add_anomalies_to_training(selected_anomalies, dest_path):

    for path in selected_anomalies:
        label = Path(path).parent.name
        dest_folder = f"{dest_path}/{label}"
        shutil.copy2(path, dest_folder)


def fraction_of_anomalies(labels_np, selected_anomalies, anomaly_label):

    total_anomalies = sum(labels_np)
    if total_anomalies == 0:
        return 0

    num_anomalies_detected = 0
    for path in selected_anomalies:
        _label = Path(path).parent.name
        if _label == anomaly_label:
            num_anomalies_detected += 1

    return num_anomalies_detected / total_anomalies


def iforest_most_anomalous(scores, test_paths, num_to_display, anomaly_label):
    accurate_score_decision, accurate_anomaly_path = [], []

    _anomalous_probaba, _anomalous_paths = zip(*sorted(zip(scores, test_paths), reverse=True))

    selected_test_path, selected_score_decision = (
        _anomalous_paths[:num_to_display],
        _anomalous_probaba[:num_to_display],
    )

    for path, decision in zip(selected_test_path, selected_score_decision):
        _label = Path(path).parent.name
        if _label == anomaly_label:
            accurate_anomaly_path.append(path)
            accurate_score_decision.append(decision)

    return accurate_anomaly_path, accurate_score_decision


def iforest_anomalies_detected(ds, most_anomalous):
    anomalies_detected = []
    all_samples = ds.samples
    for i in most_anomalous:
        if all_samples[i][-1] == 2:
            anomalies_detected.append(all_samples[i][0])
    return anomalies_detected


def test_paths(pth):
    fld = []
    for f in Path(pth).iterdir():
        if f.is_dir():
            fld.append(f)
    return os_sorted(fld)


def cp_to_init(src, dest):
    "Copies content from the first epoch in testing folder to training"
    path_src = Path(src)
    first_folder = []
    for f in path_src.iterdir():
        if f.is_dir() and f.name.split("_")[-1] == "0":
            first_folder.append(f.name)
    full_src = f"{src}/{first_folder[0]}"
    shutil.copytree(full_src, dest, dirs_exist_ok=True)


def get_np_imgs(ds):
    lst_imgs = []
    lst_labels = []
    lst_path = []

    for pth, lbl in ds.imgs:
        img = Image.open(pth)
        np_img = np.array(img).flatten()
        lst_imgs.append(np_img)
        lst_labels.append(lbl)
        lst_path.append(pth)

    np_imgs = np.array(lst_imgs)
    labels = np.array(lst_labels)

    return np_imgs, labels, lst_path


def get_np_imgs_all(train_ds, test_ds):
    lst_imgs = []
    lst_labels = []
    lst_path = []

    for pth, lbl in train_ds.imgs:
        img = Image.open(pth)
        np_img = np.array(img).flatten()
        lst_imgs.append(np_img)
        lst_labels.append(lbl)
        lst_path.append(pth)

    for pth, lbl in test_ds.imgs:
        img = Image.open(pth)
        np_img = np.array(img).flatten()
        lst_imgs.append(np_img)
        lst_labels.append(lbl)
        lst_path.append(pth)

    np_imgs = np.array(lst_imgs)
    labels = np.array(lst_labels)

    return np_imgs, labels, lst_path


def get_np_combined(train_ds, test_ds):
    lst_imgs = []
    lst_labels = []
    lst_path = []

    for pth, lbl in test_ds.imgs:
        img = Image.open(pth)
        np_img = np.array(img).flatten()
        lst_imgs.append(np_img)
        lst_labels.append(lbl)
        lst_path.append(pth)

    for pth, lbl in train_ds.imgs:
        img = Image.open(pth)
        np_img = np.array(img).flatten()
        lst_imgs.append(np_img)
        lst_labels.append(lbl)
        lst_path.append(pth)

    np_imgs = np.array(lst_imgs)
    labels = np.array(lst_labels)

    return np_imgs, labels, lst_path


def basic_reset_galaxy(_std, initial_training_config, initial_data_config, rounds, anomaly_label, add_noise):
    var = _std ** 2

    all_data_config = get_basic_config(rounds, initial_data_config, anomaly_label)

    src_original = "data/galaxy_zoo/g_zoo"
    dest_pool = "data/galaxy_zoo/pool"
    galaxy_test = "data/galaxy_zoo/testing"
    src = "data/galaxy_zoo/pool"
    galaxy_main = "data/galaxy_zoo"

    zero_folders(
        "data/galaxy_zoo/pool",
        "data/galaxy_zoo/testing",
        "data/galaxy_zoo/training",
        "data/galaxy_zoo/testing_main",
        src=src_original,
        dest=dest_pool,
    )

    training_path = "data/galaxy_zoo/training/"
    initial_training(
        "data/galaxy_zoo/pool",
        training_path,
        initial_training_config,
        var=var,
        add_noise=add_noise,
    )

    create_ds_all_config(galaxy_test, src, all_data_config, var=var, add_noise=add_noise)
    create_test_folders(initial_data_config, galaxy_main)


def basic_reset_medmnist(_std, initial_training_config, initial_data_config, rounds, anomaly_label, add_noise):
    var = _std ** 2

    all_data_config = get_basic_config(rounds, initial_data_config, anomaly_label)

    src_original = "data/medmnist/medical-mnist"
    dest_pool = "data/medmnist/pool"
    medmnist_test = "data/medmnist/testing"
    src = "data/medmnist/pool"
    medmnist_main = "data/medmnist"

    zero_folders(
        "data/medmnist/pool",
        "data/medmnist/testing",
        "data/medmnist/training",
        "data/medmnist/testing_main",
        src=src_original,
        dest=dest_pool,
    )

    training_path = "data/medmnist/training/"
    initial_training(
        "data/medmnist/pool",
        training_path,
        initial_training_config,
        var=var,
        add_noise=add_noise,
    )

    create_ds_all_config(medmnist_test, src, all_data_config, var=var, add_noise=add_noise)
    create_test_folders(initial_data_config, medmnist_main)


def basic_reset_mnist(_std, initial_training_config, initial_data_config, rounds, anomaly_label, add_noise):
    var = _std ** 2

    all_data_config = get_basic_config(rounds, initial_data_config, anomaly_label)

    src_original = "data/mnist/mnist/training"
    dest_pool = "data/mnist/pool"
    mnist_test = "data/mnist/testing"
    src = "data/mnist/pool"
    mnist_main = "data/mnist"

    zero_folders(
        "data/mnist/pool",
        "data/mnist/testing",
        "data/mnist/training",
        "data/mnist/testing_main",
        src=src_original,
        dest=dest_pool,
    )

    training_path = "data/mnist/training/"

    initial_training(
        "data/mnist/pool",
        training_path,
        initial_training_config,
        var=var,
        add_noise=add_noise,
    )

    create_ds_all_config(mnist_test, src, all_data_config, var=var, add_noise=add_noise)
    create_test_folders(initial_data_config, mnist_main)


def get_basic_config(rounds, data_config, anomaly_label, constant_anomaly_size=True):
    all_data_config = []
    for i in range(1, rounds + 1):
        if not constant_anomaly_size:
            data = {a: (int((v * i) / 2) if a == anomaly_label else v) for a, v in data_config.items()}
        else:
            data = data_config
        # if i == 1:
        #     data = {a: (0 if a == anomaly_label else v) for a, v in data_config.items()}
        all_data_config.append(data)
    return all_data_config


def initialize_pool(src, dest):
    """Initialize the data Pool. Take data from source to the pool"""
    shutil.rmtree(dest)
    pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)


def rm_folders(*args):
    """Empty Directories"""
    for arg in args:
        dir_path = pathlib.Path(arg)
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)
        pathlib.Path(arg).mkdir(parents=True, exist_ok=True)


def zero_folders(*args, src, dest):
    """Remove folders specifies in args and copy data from source to pool(dest)"""
    rm_folders(*args)

    initialize_pool(src, dest)


def load_model(arch, path):
    """Load saved model"""
    network = arch()
    network.load_state_dict(torch.load(path))
    return network


def load_optim(optim, model, path, learning_rate, momentum):
    """Load Optimizer"""
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer.load_state_dict(path)
    return optimizer


def get_transforms():
    import torchvision.transforms as transforms

    _transform = transforms.Compose(
        [
            # transforms.RandomApply(
            #     [
            #         transforms.ColorJitter(),
            #         transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
            #         # transforms.RandomErasing(),
            #     ],
            #     p=0.3,
            # ),
            transforms.RandomAffine(10, scale=(0.8, 1.2)),
            # transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    _test_transform = transforms.Compose(
        [
            # transforms.Resize((128, 128)),
            transforms.RandomAffine(10, scale=(0.8, 1.2)),
            transforms.ToTensor(),
        ]
    )
    return _transform, _test_transform


def prefix_files_with_labels(pool_path):
    p = Path(pool_path)

    folders = [f for f in p.iterdir() if f.is_dir() and f.name != "anomaly" and not f.name.startswith(".")]

    for folder in folders:
        prefix = folder.name
        files = os.listdir(folder)

        for fl in files:
            os.rename(f"{folder}/{fl}", f"{folder}/{prefix}_{fl}")


def ahunt_most_anomalous(fl_paths, anomaly_pred_probaba, anomaly_label, num_to_display):
    selected_anomalous_probaba, selected_anomalous_paths = [], []

    _anomalous_probaba, _anomalous_paths = zip(*sorted(zip(anomaly_pred_probaba, fl_paths), reverse=True))

    _selected_anomalous_probaba, _selected_anomalous_paths = (
        _anomalous_probaba[:num_to_display],
        _anomalous_paths[:num_to_display],
    )

    for path, prob in zip(_selected_anomalous_paths, _selected_anomalous_probaba):
        _label = Path(path).parent.name
        if _label == anomaly_label:
            selected_anomalous_paths.append(path)
            selected_anomalous_probaba.append(prob)

    return selected_anomalous_probaba, selected_anomalous_paths


def balance_classes(ds):
    from sklearn.utils.class_weight import compute_class_weight
    from torch.utils.data import WeightedRandomSampler

    target = np.array(ds.targets)
    cls_weights = torch.from_numpy(compute_class_weight("balanced", np.unique(target), target))
    weights = cls_weights[torch.from_numpy(target)]
    sampler = WeightedRandomSampler(weights, len(target), replacement=True)
    return sampler


def create_label_folders(img_folder_path, labels):
    "Creates Folders for Labels"
    from pathlib import Path

    folders = ["train", "test"]

    for folder in folders:
        for label in labels:
            Path(f"{img_folder_path}/{folder}/{label}").mkdir(parents=True, exist_ok=True)


def get_all_folder_files(path):
    import os

    files = [
        os.path.join(str(path), f)
        for f in os.listdir(path)
        if f.endswith((".jpg", ".JPG", ".png", ".PNG", ".JPEG", "jpeg"))
    ]
    return files


def get_file_split(files, split=0.8):

    import numpy as np

    train_split = np.random.choice(files, int(len(files) * split), replace=False).tolist()
    test_split = list(set(files) - set(train_split))

    return train_split, test_split


def create_train_and_test(src, labels, split=0.8):
    import collections

    split_dict = collections.defaultdict(list)

    for label in labels:
        label_files = get_all_folder_files(f"{src}/{label}")
        train_split, test_split = get_file_split(label_files)
        split_dict[label] = [train_split, test_split]

    return split_dict


def get_folders_in_dir(path):
    from glob import glob

    all_folders = glob(f"{path}/**/**/")
    return all_folders


def move_train_test_split(split_dict, dest):
    flds = get_folders_in_dir(dest)
    for fld in flds:
        fld_path = Path(fld)
        if fld_path.parent.name == "train":
            label_name = fld_path.name
            training_set = split_dict[label_name][0]
            cp_files_from_path(training_set, fld_path)
        if fld_path.parent.name == "test":
            label_name = fld_path.name
            test_set = split_dict[label_name][1]
            cp_files_from_path(test_set, fld_path)


def cp_files_from_path(file_paths, dest):
    import shutil

    for fl in file_paths:
        shutil.copy(fl, dest)


def evaluate_model(model, test_loader):
    actuals, predictions = torch.tensor([]), torch.tensor([])
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, (data, target, paths) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            preds = output.argmax(dim=1).to("cpu")
            predictions = torch.cat((predictions, preds), dim=0)
            actuals = torch.cat((actuals, target.to("cpu")), dim=0)
    return actuals, predictions


def combine_classes(pool_path, labels, anomaly_label, percentage):

    # create anomaly class
    anomaly_path = f"{pool_path}/{anomaly_label}"
    pathlib.Path(anomaly_path).mkdir(parents=True, exist_ok=True)

    for label in labels:
        label_src = f"{pool_path}/{label}"
        _, fl_names = get_random_fls(label_src, percentage=percentage)
        cp_files(label_src, anomaly_path, fl_names)


def prefix_files_with_labels(pool_path):
    p = Path(pool_path)

    folders = [f for f in p.iterdir() if f.is_dir() and f.name != "anomaly" and not f.name.startswith(".")]

    for folder in folders:
        prefix = folder.name
        files = os.listdir(folder)

        for fl in files:
            os.rename(f"{folder}/{fl}", f"{folder}/{prefix}_{fl}")


def get_transforms_mnist():
    import torchvision.transforms as transforms

    _transform = transforms.Compose(
        [
            transforms.RandomAffine(10, scale=(0.8, 1.2)),
            transforms.ToTensor(),
        ]
    )
    _test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return _transform, _test_transform


def mnist_parameters(anomaly_label, Net2, mnist_params):
    collective_test_path = "data/mnist/testing_main/ahunt"
    training_path = "data/mnist/training"
    mnist_test_path = "data/mnist/testing"

    train_latent_path = f"{training_path}/latent"
    train_ahunt_path = f"{training_path}/ahunt"
    train_iforest_path = f"{training_path}/iforest"

    saved_model_path = "data/mnist/results/model.pth"

    n_epoch = mnist_params["n_epoch"]
    batch_size_train = 64
    batch_size_test = 32
    learning_rate = 0.01
    momentum = 0.5

    iforest_use_all = True
    use_cummulative_test = True

    loss = mnist_params["loss"]
    query_strategy = mnist_params["query_strategy"]
    custom_model = mnist_params["custom_model"]
    train_once = mnist_params["train_once"]

    _transforms, _test_transform = get_transforms_mnist()
    test_pth = test_paths(mnist_test_path)

    classifier = Net2

    return (
        test_pth,
        _transforms,
        iforest_use_all,
        train_iforest_path,
        train_latent_path,
        train_ahunt_path,
        batch_size_train,
        batch_size_test,
        n_epoch,
        classifier,
        learning_rate,
        momentum,
        saved_model_path,
        anomaly_label,
        collective_test_path,
        use_cummulative_test,
        loss,
        query_strategy,
        mnist_test_path,
        custom_model,
        train_once,
    )


def cifar_parameters(anomaly_label, classifier, cifar_params):
    collective_test_path = "data/cifar/testing_main/ahunt"
    training_path = "data/cifar/training"
    cifar_test_path = "data/cifar/testing"

    train_latent_path = f"{training_path}/latent"
    train_ahunt_path = f"{training_path}/ahunt"
    train_iforest_path = f"{training_path}/iforest"

    saved_model_path = "data/cifar/results/model.pth"

    n_epoch = cifar_params["n_epoch"]
    batch_size_train = 64
    batch_size_test = 32
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    num_to_display = 6

    iforest_use_all = True
    use_cummulative_test = True

    loss = cifar_params["loss"]
    query_strategy = cifar_params["query_strategy"]
    custom_model = cifar_params["custom_model"]
    train_once = cifar_params["train_once"]

    _transforms, _test_transform = get_transforms_mnist()
    test_pth = test_paths(cifar_test_path)

    return (
        test_pth,
        _transforms,
        iforest_use_all,
        train_iforest_path,
        train_latent_path,
        train_ahunt_path,
        batch_size_train,
        batch_size_test,
        n_epoch,
        classifier,
        learning_rate,
        momentum,
        saved_model_path,
        anomaly_label,
        collective_test_path,
        use_cummulative_test,
        loss,
        query_strategy,
        cifar_test_path,
        custom_model,
        train_once,
    )
