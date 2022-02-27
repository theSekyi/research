import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import numpy as np
from pandas.core.common import flatten

# from focal_loss.focal_loss import FocalLoss


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import torch.nn as nn


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


class IsoMaxPlusLossFirstPart(nn.Module):
    """Replaces the model classifier last layer nn.Linear()"""

    def __init__(self, num_features, num_classes):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1))
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        # print("isomax plus loss first part")
        distances = F.pairwise_distance(
            F.normalize(features).unsqueeze(2), F.normalize(self.prototypes).t().unsqueeze(0), p=2.0
        )
        logits = -torch.abs(self.distance_scale) * distances
        return logits


class IsoMaxPlusLossSecondPart(nn.Module):
    """Replaces the nn.CrossEntropyLoss()"""

    def __init__(self, model_classifier):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.model_classifier = model_classifier
        self.entropic_scale = 10.0

    def forward(self, logits, targets, debug=False):
        ################################################################################
        ################################################################################
        """Probabilities and logarithms are calculate separately and sequentially!!!"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss!!!"""
        ################################################################################
        ################################################################################
        # print("isomax plus loss second part")
        distance_scale = torch.abs(self.model_classifier.distance_scale)
        probabilities_for_training = nn.Softmax(dim=1)(self.entropic_scale * logits[: len(targets)])
        probabilities_at_targets = probabilities_for_training[range(logits.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        if not debug:
            return loss
        else:
            targets_one_hot = torch.eye(self.model_classifier.prototypes.size(0))[targets].long().cuda()
            intra_inter_logits = torch.where(
                targets_one_hot != 0, -logits[: len(targets)], torch.Tensor([float("Inf")]).cuda()
            )
            inter_intra_logits = torch.where(
                targets_one_hot != 0, torch.Tensor([float("Inf")]).cuda(), -logits[: len(targets)]
            )
            intra_logits = intra_inter_logits[intra_inter_logits != float("Inf")]
            inter_logits = inter_intra_logits[inter_intra_logits != float("Inf")]
            return loss, distance_scale.item(), intra_logits, inter_logits


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def test(network, test_loader, anomaly_label, anomaly_idx, index_separator, device):

    label = []
    fl_paths = []
    scores_decision = []
    prob_difference = []
    actual_labels = []
    predicted_labels = []
    all_probs = torch.tensor([])
    all_outputs = torch.tensor([])

    network.eval()
    network.to(device)

    with torch.no_grad():
        for i, (data, target, paths) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = network(data)
            probability = F.softmax(output, dim=1)  # get probabilities
            actual_labels.extend(target.tolist())
            all_probs = torch.cat((all_probs, probability.cpu()), dim=0)
            all_outputs = torch.cat((all_outputs, output.cpu()), dim=0)
            #             probs.extend(probability)

            _, predicted = torch.max(output.data, 1)
            predicted_labels.extend(predicted.tolist())

            _score_decision = probability[:, anomaly_idx]  # get probability of anomalous class
            normal_index_sum = torch.sum(probability[:, :index_separator], dim=1, keepdim=True)
            anomaly_index_sum = torch.sum(probability[:, index_separator:], dim=1, keepdim=True)
            difference = abs(normal_index_sum - anomaly_index_sum).reshape(-1).tolist()
            prob_difference.extend(difference)

            _labels = [Path(lbl).parent.name for lbl in paths]

            _paths = list(paths)
            scores_decision.append(_score_decision.tolist())

            label.append(_labels)
            fl_paths.append(_paths)

    label_flat = list(flatten(label))

    labels_np = np.array([True if x == anomaly_label else False for x in label_flat])
    fl_paths = list(flatten(fl_paths))
    scores_decision = list(flatten(scores_decision))

    return (
        fl_paths,
        labels_np,
        scores_decision,
        prob_difference,
        actual_labels,
        predicted_labels,
        all_probs,
        all_outputs,
    )


def test_classifer(network, test_loader, anomaly_label, anomaly_idx, device):

    label = []
    fl_paths = []
    scores_decision = []
    predictions = []

    network.eval()
    network.to(device)

    with torch.no_grad():
        for i, (data, target, paths) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = network(data)

            _output = F.softmax(output.data, dim=1)  # get probabilities

            _, predicted = torch.max(output.data, 1)

            predictions.append(predicted.tolist())

            _score_decision = _output[:, anomaly_idx]  # get probability of anomalous class

            _labels = [Path(lbl).parent.name for lbl in paths]

            _paths = list(paths)
            scores_decision.append(_score_decision.tolist())

            label.append(_labels)
            fl_paths.append(_paths)

        label_flat = list(flatten(label))
        pred_flat = list(flatten(predictions))

        labels_np = np.array([True if x == anomaly_label else False for x in label_flat])
        y_pred = np.array([True if x == anomaly_label else False for x in pred_flat])

        fl_paths = list(flatten(fl_paths))
        scores_decision = list(flatten(scores_decision))

    return fl_paths, labels_np, y_pred, scores_decision


def train(
    epoch,
    optimizer,
    network,
    train_loader,
    saved_model_path,
    criterion,
    device,
):
    network.to(device)
    network.train()

    n_batches = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        train_losses = loss.item()
        print(f"\t Batch {batch_idx + 1}/{n_batches} | Training Loss {train_losses}")

    torch.save(network.state_dict(), saved_model_path)
    torch.save(optimizer.state_dict(), "data/mnist/results/optimizer.pth")
    return train_losses, network


def get_network_and_optimizer(Net, optimizer, learning_rate, momentum):
    network_type = Net.__class__.__name__  # get name of architecture. Please dont do this. This a hack

    if network_type == "ResNet":
        network = Net
    else:
        network = Net()

    optimizer = optimizer.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    return network, optimizer


def training_loop(
    n_epoch,
    optimizer,
    Net,
    train_loader,
    learning_rate,
    momentum,
    saved_model_path,
    train_func,
    loss,
    device,
):
    network, optimizer = get_network_and_optimizer(Net, optimizer, learning_rate, momentum)

    if loss == "focal":
        criterion = FocalLoss()
    elif loss == "iso":
        criterion = IsoMaxPlusLossSecondPart(network.fc)
    elif loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()  # F.nll_loss

    train_losses = []
    for epoch in range(1, n_epoch):
        print(f"{color.BOLD}Epoch {epoch}/{n_epoch - 1} {color.END}")
        train_loss, model = train_func(
            epoch,
            optimizer,
            network,
            train_loader,
            saved_model_path,
            criterion,
            device,
        )
        train_losses.append(train_loss)
    return train_losses, model
