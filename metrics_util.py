import torch

def acc(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true)\
        .type(torch.float32).mean().item()


def f1(output, target):
    output = torch.max(output,1).indices
    tp = (target * output).sum().to(torch.float32)
    tn = ((1 - target) * (1 - output)).sum().to(torch.float32)
    fp = ((1 - target) * output).sum().to(torch.float32)
    fn = (target * (1 - output)).sum().to(torch.float32)
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return [precision.item(), recall.item(), f1.item()]
