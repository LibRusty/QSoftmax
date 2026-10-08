import os
import json
import argparse
import numpy as np

import qsoftmax_pipeline as qsm

FIXED_Q_RESULTS_FILE = "./fixed_q_results.json"


def _load(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _noise_tag(noise, eta):
    return f"synthetic{eta:.2f}" if noise == "synthetic" else noise


def _fixed_key(dataset, model, noise, eta, seed, q):
    return f"fixedq_{dataset}_{model}_{_noise_tag(noise, eta)}_seed{seed}_q{q:.4f}"


def _ce_key(dataset, model, noise, eta, seed):
    return f"qsm_{dataset}_{model}_{_noise_tag(noise, eta)}_seed{seed}"


def run_one(dataset, noise, seed, model, q, eta=0.4):
    qsm.configure(dataset=dataset, noise_type=noise, seed=seed, n_trials=0,
                  model_name=model, synthetic_eta=eta)

    key = _fixed_key(dataset, model, noise, eta, seed, q)
    results = _load(FIXED_Q_RESULTS_FILE)
    if key in results:
        r = results[key]
        qsm.notify(f"Fixed-q результат для {key} уже есть: val={r['val_acc']:.2f}%, "
                   f"test={r['test_acc']:.2f}% (пересчёт пропущен)")
        return r["val_acc"], r["test_acc"]

    qsm.notify(f"Обучаю с фиксированным q={q}: {key}")
    train_loader, val_loader, test_loader = qsm.get_dataloaders()
    criterion = qsm.QSoftmaxLoss(q=q, label_smoothing=qsm.LABEL_SMOOTHING)
    val_acc, test_acc = qsm.train_run(train_loader, val_loader, test_loader, criterion, run_tag=key)
    qsm.notify(f"Fixed-q {key}: val={val_acc:.2f}%, test={test_acc:.2f}%")

    results[key] = {"val_acc": val_acc, "test_acc": test_acc, "q": q}
    _save(FIXED_Q_RESULTS_FILE, results)
    return val_acc, test_acc


def summarize(dataset, noise, model, q, seeds, eta=0.4):
    fixed_results = _load(FIXED_Q_RESULTS_FILE)
    ce_results = _load(qsm.BASELINE_RESULTS_FILE)

    ce_vals, fixed_vals = [], []
    print(f"{'seed':>6} | {'CE test':>10} | {'FixedQ test':>12}")
    print("-" * 34)
    for seed in seeds:
        ce_entry = ce_results.get(_ce_key(dataset, model, noise, eta, seed))
        fq_entry = fixed_results.get(_fixed_key(dataset, model, noise, eta, seed, q))

        ce_test = ce_entry["test_acc"] if ce_entry else None
        fq_test = fq_entry["test_acc"] if fq_entry else None

        print(f"{seed:>6} | {f'{ce_test:.2f}%' if ce_test is not None else 'нет данных':>10} | "
              f"{f'{fq_test:.2f}%' if fq_test is not None else 'нет данных':>12}")

        if ce_test is not None:
            ce_vals.append(ce_test)
        if fq_test is not None:
            fixed_vals.append(fq_test)

    print()
    if ce_vals:
        print(f"CE:      mean={np.mean(ce_vals):.2f}%  std={np.std(ce_vals):.2f}  (n={len(ce_vals)})")
    else:
        print("CE: данных нет (сначала прогоните qsoftmax_pipeline.py для этих сидов)")
    if fixed_vals:
        print(f"FixedQ:  mean={np.mean(fixed_vals):.2f}%  std={np.std(fixed_vals):.2f}  (n={len(fixed_vals)}, q={q})")
    else:
        print("FixedQ: данных нет")

    if len(ce_vals) == len(fixed_vals) == len(seeds) and seeds:
        diff = np.array(fixed_vals) - np.array(ce_vals)
        print(f"\nРазница (FixedQ - CE) по сидам: {np.round(diff, 2).tolist()}")
        print(f"Средняя разница: {diff.mean():+.2f} п.п., std={diff.std():.2f}")


def main():
    parser = argparse.ArgumentParser(description="Обучение с фиксированным q (без Optuna), по списку сидов.")
    parser.add_argument("--dataset", required=True, choices=list(qsm.DATASET_SPECS))
    parser.add_argument("--noise", required=True)
    parser.add_argument("--eta", type=float, default=0.4)
    parser.add_argument("--model", default="resnet34", choices=list(qsm.MODEL_BUILDERS))
    parser.add_argument("--q", type=float, required=True, help="Фиксированное значение q (например, лучшее из Optuna)")
    parser.add_argument("--seed", type=int, help="Один сид (для параллельного запуска нескольких процессов из .sh)")
    parser.add_argument("--seeds", type=str, help="Список сидов через запятую: 0,1,42,123,777")
    parser.add_argument("--summary-only", action="store_true",
                         help="Только напечатать сравнение CE vs FixedQ по уже посчитанным сидам, без обучения")
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        parser.error("Укажите --seed (один) или --seeds (список через запятую)")

    if args.summary_only:
        summarize(args.dataset, args.noise, args.model, args.q, seeds, eta=args.eta)
        return

    for seed in seeds:
        run_one(args.dataset, args.noise, seed, args.model, args.q, eta=args.eta)

    if len(seeds) > 1:
        print()
        summarize(args.dataset, args.noise, args.model, args.q, seeds, eta=args.eta)


if __name__ == "__main__":
    main()
