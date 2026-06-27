import os
import csv
import time
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report

SEED = 123

BATCH_SIZE = 128
EPOCHS = 200

LR = 0.1
WEIGHT_DECAY = 5e-4

NUM_WORKERS = 4

QS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

TEMPERATURES = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0
]

LOG_DIR = Path("logs")
RESULTS_DIR = Path("results")

LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

def create_logger(q, T):

    logger = logging.getLogger(f"q_{q:.2f}_T_{T:.2f}")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    log_path = LOG_DIR / f"q_{q:.2f}_T_{T:.2f}.log"

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def load_data():

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader


def q_softmax(logits, q):

    if abs(q - 1.0) < 1e-6:
        return torch.softmax(logits, dim=1)

    base = 1.0 + (1.0 - q) * logits
    base = torch.clamp(base, min=1e-8)

    exps = torch.pow(base, 1.0 / (1.0 - q))

    probs = exps / (exps.sum(dim=1, keepdim=True) + 1e-8)

    return probs

def q_cross_entropy_from_logits(logits, labels, q, T=1.0):

    logits = logits / T
    logits = torch.clamp(logits, -10, 10)

    if abs(q - 1.0) < 1e-6:
        return nn.functional.cross_entropy(logits, labels)

    probs = q_softmax(logits, q)

    idx = torch.arange(labels.size(0), device=labels.device)

    p = torch.clamp(probs[idx, labels], min=1e-8)

    loss = -((torch.pow(p, 1.0 - q) - 1.0) / (1.0 - q))

    return loss.mean()


class EarlyStopping:

    def __init__(self, patience=15):

        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, value):

        if value < self.best:

            self.best = value
            self.counter = 0

            return False

        self.counter += 1

        return self.counter >= self.patience


def get_model():

    model = torchvision.models.resnet18(num_classes=10)

    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()

    return model


def eval_loss(model, loader, q, T):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x)

            loss = q_cross_entropy_from_logits(
                logits,
                y,
                q=q,
                T=T
            )

            total_loss += loss.item()

    return total_loss / len(loader)

def train(model, train_loader, val_loader, q, T, logger):

    model.to(DEVICE)

    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    early_stopping = EarlyStopping(patience=15)

    best_val = float("inf")

    for epoch in range(EPOCHS):

        start_time = time.time()

        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            logits = model(x)

            loss = q_cross_entropy_from_logits(
                logits,
                y,
                q=q,
                T=T
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():

                probs = q_softmax(logits / T, q)

                pred = probs.argmax(dim=1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        scheduler.step()

        train_loss /= len(train_loader)

        train_acc = 100.0 * correct / total

        val_loss = eval_loss(
            model,
            val_loader,
            q=q,
            T=T
        )

        lr_now = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - start_time

        logger.info(
            f"epoch={epoch+1:03d} | "
            f"lr={lr_now:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.2f} | "
            f"val_loss={val_loss:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss

        if early_stopping.step(val_loss):

            logger.info(
                f"early stopping at epoch {epoch+1}"
            )

            break


def evaluate(model, loader, q, T, logger):

    model.eval()

    preds = []
    labels_all = []

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x) / T

            probs = q_softmax(logits, q)

            pred = probs.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            preds.extend(pred.cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    acc = 100.0 * correct / total

    logger.info("")
    logger.info(f"FINAL TEST ACCURACY = {acc:.2f}%")
    logger.info("")
    logger.info(
        "\n" + classification_report(
            labels_all,
            preds,
            digits=4
        )
    )

    return acc


def run_experiment(train_loader,
                   val_loader,
                   test_loader,
                   q,
                   T):

    logger = create_logger(q, T)

    logger.info("=" * 80)
    logger.info(f"START EXPERIMENT: q={q:.2f} | T={T:.2f}")
    logger.info("=" * 80)

    model = get_model()

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        q=q,
        T=T,
        logger=logger
    )

    acc = evaluate(
        model=model,
        loader=test_loader,
        q=q,
        T=T,
        logger=logger
    )

    logger.info(
        f"END EXPERIMENT: q={q:.2f} | T={T:.2f} | acc={acc:.2f}"
    )

    return acc


def save_results(results):

    csv_path = RESULTS_DIR / "results.csv"

    results = sorted(
        results,
        key=lambda x: x["accuracy"],
        reverse=True
    )

    with open(csv_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "q",
                "temperature",
                "accuracy"
            ]
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)


def main():

    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader = load_data()

    results = []

    total_runs = len(QS) * len(TEMPERATURES)
    run_idx = 0

    for T in TEMPERATURES:

        for q in QS:

            run_idx += 1

            print("\n" + "=" * 80)
            print(
                f"RUN {run_idx}/{total_runs} | "
                f"q={q:.2f} | T={T:.2f}"
            )
            print("=" * 80)

            acc = run_experiment(
                train_loader,
                val_loader,
                test_loader,
                q,
                T
            )

            results.append({
                "q": q,
                "temperature": T,
                "accuracy": round(acc, 4)
            })

            save_results(results)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    results = sorted(
        results,
        key=lambda x: x["accuracy"],
        reverse=True
    )

    for r in results:

        print(
            f"q={r['q']:.2f} | "
            f"T={r['temperature']:.2f} | "
            f"acc={r['accuracy']:.2f}"
        )

    print("\nLogs saved to: ./logs/")
    print("Results saved to: ./results/results.csv")



if __name__ == "__main__":
    main()