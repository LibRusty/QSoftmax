import os
import json
import time
import copy
import random
import argparse
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet18, resnet34
import optuna

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 80
LR = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.0
NUM_WORKERS = 2
CUTOUT_SIZE = 8
DATA_DIR = "./data"

MODEL_BUILDERS = {"resnet18": resnet18, "resnet34": resnet34}

DATASET_SPECS = {
    "cifar10": {
        "torchvision_cls": torchvision.datasets.CIFAR10,
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "label_url": "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-10_human.pt",
        "label_filename": "CIFAR-10_human.pt",
        "noise_keys": {
            "clean": "clean_label", "aggre": "aggre_label",
            "random1": "random_label1", "random2": "random_label2", "random3": "random_label3",
            "worst": "worse_label",
        },
    },
    "cifar100": {
        "torchvision_cls": torchvision.datasets.CIFAR100,
        "num_classes": 100,
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
        "label_url": "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-100_human.pt",
        "label_filename": "CIFAR-100_human.pt",
        "noise_keys": {"clean": "clean_label", "noisy": "noisy_label"},
    },
}

# ------------------------- Параметры текущего запуска -------------------------
DATASET = NOISE_TYPE = SYNTHETIC_ETA = SEED = N_TRIALS = MODEL_NAME = None
Q_MIN = Q_MAX = VAL_FRACTION = NUM_CLASSES = None
RUN_TAG = LOG_FILE = STUDY_DB_PATH = STUDY_NAME = SPEC = None
BASELINE_RESULTS_FILE = "./ce_baseline_results_qsoftmax.json"


def configure(dataset, noise_type, seed, n_trials, model_name,
              synthetic_eta=0.4, q_min=0.3, q_max=2.0, val_fraction=0.1):
    global DATASET, NOISE_TYPE, SYNTHETIC_ETA, SEED, N_TRIALS, MODEL_NAME
    global Q_MIN, Q_MAX, VAL_FRACTION, NUM_CLASSES, RUN_TAG, LOG_FILE
    global STUDY_DB_PATH, STUDY_NAME, SPEC

    assert dataset in DATASET_SPECS, f"--dataset должен быть одним из {list(DATASET_SPECS)}"
    spec = DATASET_SPECS[dataset]
    allowed_noise = list(spec["noise_keys"]) + ["synthetic"]
    assert noise_type in allowed_noise, f"--noise для {dataset} должен быть одним из {allowed_noise}"
    assert model_name in MODEL_BUILDERS, f"--model должен быть одним из {list(MODEL_BUILDERS)}"

    DATASET, SPEC, NOISE_TYPE, SYNTHETIC_ETA = dataset, spec, noise_type, synthetic_eta
    SEED, N_TRIALS, MODEL_NAME = seed, n_trials, model_name
    Q_MIN, Q_MAX, VAL_FRACTION = q_min, q_max, val_fraction
    NUM_CLASSES = spec["num_classes"]

    noise_tag = f"synthetic{SYNTHETIC_ETA:.2f}" if NOISE_TYPE == "synthetic" else NOISE_TYPE
    RUN_TAG = f"qsm_{DATASET}_{MODEL_NAME}_{noise_tag}_seed{SEED}"
    LOG_FILE = f"./log_{RUN_TAG}.txt"
    STUDY_DB_PATH = f"sqlite:///./optuna_{RUN_TAG}.db"
    STUDY_NAME = f"qsoftmax_{RUN_TAG}"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def notify(msg):
    print(msg, flush=True)
    log(msg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def download_noise_labels():
    os.makedirs(DATA_DIR, exist_ok=True)
    label_path = os.path.join(DATA_DIR, SPEC["label_filename"])
    if os.path.exists(label_path):
        log(f"Файл меток уже есть: {label_path}")
        return label_path
    log(f"Скачиваю метки {SPEC['label_filename']}...")
    urllib.request.urlretrieve(SPEC["label_url"], label_path)
    log("Готово.")
    return label_path


class Cutout:
    def __init__(self, size=8):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()
        y1, y2 = max(0, y - self.size // 2), min(h, y + self.size // 2)
        x1, x2 = max(0, x - self.size // 2), min(w, x + self.size // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img


class NoisyDataset(Dataset):
    def __init__(self, base_dataset, noisy_labels):
        self.base = base_dataset
        assert len(noisy_labels) == len(base_dataset)
        self.noisy_labels = noisy_labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _clean_label = self.base[idx]
        return image, int(self.noisy_labels[idx])


def generate_symmetric_noise(clean_labels, eta, num_classes, seed):
    rng = np.random.RandomState(seed)
    clean_labels = np.array(clean_labels)
    noisy_labels = clean_labels.copy()
    flip_mask = rng.rand(len(clean_labels)) < eta
    for idx in np.where(flip_mask)[0]:
        true_label = clean_labels[idx]
        choices = [c for c in range(num_classes) if c != true_label]
        noisy_labels[idx] = rng.choice(choices)
    return noisy_labels


def get_dataloaders():
    mean, std = SPEC["mean"], SPEC["std"]
    torchvision_cls = SPEC["torchvision_cls"]

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
        Cutout(CUTOUT_SIZE),
    ])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    base_train_aug = torchvision_cls(DATA_DIR, train=True, download=False, transform=train_tf)
    base_train_eval = torchvision_cls(DATA_DIR, train=True, download=False, transform=eval_tf)
    test_set = torchvision_cls(DATA_DIR, train=False, download=False, transform=eval_tf)

    if NOISE_TYPE == "synthetic":
        noisy_labels = generate_symmetric_noise(
            base_train_aug.targets, eta=SYNTHETIC_ETA, num_classes=NUM_CLASSES, seed=SEED
        )
        log(f"[synthetic] dataset={DATASET}, eta={SYNTHETIC_ETA}, seed={SEED}")
    else:
        label_path = download_noise_labels()
        label_dict = torch.load(label_path, weights_only=False)
        noisy_labels = label_dict[SPEC["noise_keys"][NOISE_TYPE]]

    actual_noise_rate = 100.0 * np.mean(np.array(noisy_labels) != np.array(base_train_aug.targets))
    log(f"[{DATASET}/{NOISE_TYPE}] реальный уровень расхождения с чистыми метками: {actual_noise_rate:.2f}%")

    n_total = len(base_train_aug)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_total)
    n_val = int(n_total * VAL_FRACTION)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    log(f"Train/Val split: {len(train_idx)} train / {len(val_idx)} val (val_fraction={VAL_FRACTION})")

    train_noisy = NoisyDataset(base_train_aug, noisy_labels)
    val_noisy = NoisyDataset(base_train_eval, noisy_labels)   # без аугментаций, те же шумные метки

    train_subset = Subset(train_noisy, train_idx)
    val_subset = Subset(val_noisy, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader


# ------------------------- Модель -------------------------
def build_model():
    ctor = MODEL_BUILDERS[MODEL_NAME]
    model = ctor(weights=None, num_classes=NUM_CLASSES)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ------------------------- Q-Softmax: замена активации -------------------------
class QSoftmax(nn.Module):
    def __init__(self, q=1.0, eps=1e-8):
        super().__init__()
        self.q = q
        self.eps = eps

    def forward(self, logits):
        if abs(self.q - 1.0) < 1e-6:
            return F.softmax(logits, dim=-1)
        z = logits - logits.max(dim=-1, keepdim=True).values
        base = torch.clamp(1.0 + (1.0 - self.q) * z, min=0.0)   # [.]_+
        exp_q = base.pow(1.0 / (1.0 - self.q))
        return exp_q / (exp_q.sum(dim=-1, keepdim=True) + self.eps)


class QSoftmaxLoss(nn.Module):
    def __init__(self, q=1.0, label_smoothing=0.0, eps=1e-8):
        super().__init__()
        self.q_softmax = QSoftmax(q=q, eps=eps)
        self.label_smoothing = label_smoothing
        self.eps = eps

    def forward(self, logits, targets):
        probs = self.q_softmax(logits).clamp(min=self.eps, max=1.0)
        log_probs = torch.log(probs)
        num_classes = logits.size(1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                true_dist = torch.full_like(probs, self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            return -(true_dist * log_probs).sum(dim=1).mean()
        return F.nll_loss(log_probs, targets)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ------------------------- Обучение одного прогона -------------------------
def train_run(train_loader, val_loader, test_loader, criterion, run_tag=""):
    set_seed(SEED)
    model = build_model().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                 weight_decay=WEIGHT_DECAY, nesterov=True)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        pct_start=0.3, div_factor=10, final_div_factor=100,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    log(f"[{run_tag}] старт: {EPOCHS} эпох, batch_size={BATCH_SIZE}, dataset={DATASET}, "
        f"model={MODEL_NAME}, num_classes={NUM_CLASSES}, seed={SEED}")

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        val_acc = evaluate(model, val_loader)
        dt = time.time() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        log(f"[{run_tag}] epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f}% | val_acc={val_acc:.2f}% | best_val={best_val_acc:.2f}% | {dt:.1f}s")

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader)
    log(f"[{run_tag}] готово. best_val_acc={best_val_acc:.2f}% | test_acc(на лучшем val)={test_acc:.2f}%")
    return best_val_acc, test_acc


# ------------------------- CE-бейзлайн (кэшируется) -------------------------
def _load_baseline_results():
    if os.path.exists(BASELINE_RESULTS_FILE):
        with open(BASELINE_RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_baseline_results(results):
    with open(BASELINE_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def get_or_train_ce_baseline(train_loader, val_loader, test_loader):
    results = _load_baseline_results()
    key = RUN_TAG
    if key in results:
        r = results[key]
        notify(f"CE-бейзлайн для {key} уже есть: val={r['val_acc']:.2f}%, test={r['test_acc']:.2f}% (пересчёт пропущен)")
        return r["val_acc"], r["test_acc"]

    notify(f"Обучаю CE-бейзлайн (обычный Softmax, q=1.0): {key} (подробности в {LOG_FILE})")
    criterion = QSoftmaxLoss(q=1.0, label_smoothing=LABEL_SMOOTHING)   # q=1.0 == обычный Softmax+CE
    val_acc, test_acc = train_run(train_loader, val_loader, test_loader, criterion, run_tag=f"CE-{key}")
    notify(f"CE-бейзлайн {key}: val={val_acc:.2f}%, test={test_acc:.2f}%")

    results[key] = {"val_acc": val_acc, "test_acc": test_acc}
    _save_baseline_results(results)
    return val_acc, test_acc


def get_study():
    study = optuna.create_study(study_name=STUDY_NAME, storage=STUDY_DB_PATH,
                                 direction="maximize", load_if_exists=True)
    n_done = len(study.trials)
    if n_done > 0:
        bt = study.best_trial
        notify(f"Study '{STUDY_NAME}' уже содержит {n_done} trial(ов). "
               f"Лучший по val: q={bt.params['q']:.4f} -> val={bt.value:.2f}%, "
               f"test={bt.user_attrs.get('test_acc', float('nan')):.2f}%")
    else:
        notify(f"Создана новая study '{STUDY_NAME}'.")
    return study


def run_optuna_qsoftmax(train_loader, val_loader, test_loader):
    study = get_study()

    def objective(trial):
        q = trial.suggest_float("q", Q_MIN, Q_MAX)
        log(f"\n[Trial {trial.number}] {RUN_TAG}, q={q:.4f} (все {EPOCHS} эпох)")
        t0 = time.time()
        criterion = QSoftmaxLoss(q=q, label_smoothing=LABEL_SMOOTHING)
        val_acc, test_acc = train_run(train_loader, val_loader, test_loader, criterion,
                                       run_tag=f"trial{trial.number}-q={q:.3f}")
        trial.set_user_attr("test_acc", test_acc)
        dt = time.time() - t0
        log(f"[Trial {trial.number}] q={q:.4f} -> val={val_acc:.2f}%, test={test_acc:.2f}% ({dt/60:.1f} мин)")
        return val_acc   # Optuna оптимизирует ИМЕННО val, не test — без утечки

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    bt = study.best_trial
    notify(f"\nПрогон завершён. Всего trial'ов в study: {len(study.trials)}")
    notify(f"Лучший q по val: {bt.params['q']:.4f} -> val={bt.value:.2f}%, "
           f"test={bt.user_attrs.get('test_acc', float('nan')):.2f}%")
    return study


def show_study_history(dataset, noise_type, seed, model_name, synthetic_eta=0.4):
    configure(dataset=dataset, noise_type=noise_type, seed=seed, n_trials=0, model_name=model_name,
              synthetic_eta=synthetic_eta)
    study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB_PATH)
    print(f"Study '{STUDY_NAME}': {len(study.trials)} trial(ов)\n", flush=True)
    rows = sorted(
        [(t.number, t.params.get("q"), t.value, t.user_attrs.get("test_acc"), t.state.name) for t in study.trials],
        key=lambda r: (r[2] is None, -(r[2] or 0)),
    )
    for number, q, val, test, state in rows:
        q_str = f"{q:.4f}" if q is not None else "-"
        val_str = f"{val:.2f}%" if val is not None else "-"
        test_str = f"{test:.2f}%" if test is not None else "-"
        print(f"  trial {number:3d} | q={q_str} | val={val_str} | test={test_str} | {state}", flush=True)
    return study


def run(dataset, noise_type, seed, n_trials, model_name, synthetic_eta=0.4,
        q_min=0.3, q_max=2.0, val_fraction=0.1):
    configure(dataset=dataset, noise_type=noise_type, seed=seed, n_trials=n_trials,
              model_name=model_name, synthetic_eta=synthetic_eta, q_min=q_min, q_max=q_max,
              val_fraction=val_fraction)

    eta_part = f" (eta={SYNTHETIC_ETA})" if NOISE_TYPE == "synthetic" else ""
    notify("=" * 60)
    notify(f"Q-Softmax pipeline | dataset={DATASET} | model={MODEL_NAME} | noise={NOISE_TYPE}{eta_part} "
           f"| seed={SEED} | n_trials={N_TRIALS}")
    notify("=" * 60)
    notify(f"Устройство: {DEVICE}")
    if DEVICE.type == "cuda":
        notify(f"GPU: {torch.cuda.get_device_name(0)}")
    notify(f"Подробный лог обучения пишется в: {os.path.abspath(LOG_FILE)}")

    train_loader, val_loader, test_loader = get_dataloaders()
    ce_val, ce_test = get_or_train_ce_baseline(train_loader, val_loader, test_loader)
    study = run_optuna_qsoftmax(train_loader, val_loader, test_loader)

    bt = study.best_trial
    best_test = bt.user_attrs.get("test_acc", float("nan"))
    notify("\n" + "=" * 60)
    notify(f"ИТОГ | dataset={DATASET} | model={MODEL_NAME} | noise={NOISE_TYPE}{eta_part} | seed={SEED}")
    notify("=" * 60)
    notify(f"CE (Softmax, q=1.0):  val={ce_val:.2f}%  test={ce_test:.2f}%")
    notify(f"Лучший Q-Softmax:      val={bt.value:.2f}%  test={best_test:.2f}%  (q={bt.params['q']:.4f})")
    notify(f"Разница по test:       {best_test - ce_test:+.2f} п.п.")

    return (ce_val, ce_test), study


def main():
    parser = argparse.ArgumentParser(description="Q-Softmax: CE-бейзлайн + Optuna-поиск q, train/val/test.")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_SPECS))
    parser.add_argument("--noise", type=str, required=True,
                         help="cifar10: clean/aggre/random1/random2/random3/worst/synthetic; "
                              "cifar100: clean/noisy/synthetic")
    parser.add_argument("--eta", type=float, default=0.4, help="Уровень synthetic-шума (только --noise synthetic)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=20, help="Сколько НОВЫХ optuna-trial'ов запустить")
    parser.add_argument("--model", type=str, default="resnet34", choices=list(MODEL_BUILDERS))
    parser.add_argument("--qmin", type=float, default=0.3)
    parser.add_argument("--qmax", type=float, default=2.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    run(dataset=args.dataset, noise_type=args.noise, seed=args.seed, n_trials=args.trials,
        model_name=args.model, synthetic_eta=args.eta, q_min=args.qmin, q_max=args.qmax,
        val_fraction=args.val_fraction)


if __name__ == "__main__":
    main()
