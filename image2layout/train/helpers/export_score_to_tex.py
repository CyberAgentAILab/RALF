import os
from glob import glob


def load_k_scores(ROOT):
    # Load pre-computed scores
    METRICS = [
        "alignment-LayoutGAN++",
        "overlap-LayoutGAN++",
        "utilization",
        "occlusion",
        "unreadability",
        "overlay",
        "underlay_effectiveness_loose",
        "underlay_effectiveness_strict",
        "R_{shm} (vgg distance)",
        "validity",
        "test_precision_layout",
        "test_recall_layout",
        "test_density_layout",
        "test_coverage_layout",
        "test_fid_layout",
    ]
    unique_tasks = [
        "uncond",
        "c",
        "cwh",
        "partial",
        "refinement",
        "relation",
        "relation_backtrack",
    ]
    dirs = glob(os.path.join(ROOT, "generated_samples*"))
    dirs = [d for d in dirs if not "debug" in d]
    SCORES = {}

    for d in dirs:
        task_name = d.split("/")[-1].split("_")[2]
        if "dynamictopk" in d:
            k = d.split("dynamictopk_")[-1].split("_backtrack")[0].split("_randomdb")[0]
        else:
            k = "0"

        use_backtrack = False
        if "backtrack" in d:
            use_backtrack = True
            task_name = f"{task_name}_backtrack"
        score_path = os.path.join(d, "scores_all.txt")
        if not os.path.exists(score_path) or not task_name in unique_tasks:
            print(f"Skip!! {score_path}")
            continue

        SCORES[task_name] = {}

        with open(score_path) as f:
            _score = f.readlines()
            _score = [s.strip() for s in _score]

        metric_name = _score[1:16]
        try:
            assert (
                metric_name == METRICS
            ), f"{metric_name} != {METRICS}, \ndiff={set(metric_name) - set(METRICS)})\n{ROOT=}"
        except:
            from IPython import embed

            embed()
            exit()
        test_score = _score[20:35]
        # val_score = _score[35:]
        test_score = {_k: _v for _k, _v in zip(METRICS, test_score)}
        # val_score = {_k: _v for _k, _v in zip(METRICS, val_score)}
        SCORES[task_name][k] = {
            "test": test_score,
            # "val": val_score,
        }

    return SCORES


def export_score_as_csv(ROOT):

    unique_tasks = [
        "uncond",
        "c",
        "cwh",
        "partial",
        "refinement",
        "relation_backtrack",
    ]

    METRICS = [
        "occlusion",
        "unreadability",
        "underlay_effectiveness_strict",
        "overlay",
        "test_fid_layout",
    ]
    KETA = {
        "underlay_effectiveness_strict": "{:.2f}",
        "overlay": "{:.3f}",
        "occlusion": "{:.3f}",
        "unreadability": "{:.4f}",
        "test_density_layout": "{:.2f}",
        "test_coverage_layout": "{:.2f}",
        "test_fid_layout": "{:.2f}",
    }

    SCORES = load_k_scores(ROOT)

    expname = ROOT.split("/")[-2]
    K = [0, 1, 2, 4, 8, 16]
    for split in ["test"]:
        OUT = f"{expname}\n"
        LATEX = [f"{expname}"]
        for idx, task in enumerate(unique_tasks):

            if not task in SCORES:
                continue

            _column = ["", "", "", ""] + METRICS
            _column = ",".join(_column) + "\n"
            OUT += _column
            LATEX += [task] + METRICS + ["\n"]
            for k in K:

                if not str(k) in SCORES[task]:
                    continue

                _c = []
                if idx == 0:
                    _c += [f"k={k}"]
                else:
                    _c += [""]
                _c += [task]
                _c += ["✔︎"]
                _c += [""]
                _out = _c
                # _out += list(map(str, SCORES[task][str(k)][split]))
                try:
                    _out += [SCORES[task][str(k)][split][m] for m in METRICS]
                except:
                    from IPython import embed

                    embed()
                    exit()
                OUT += ",".join(_out) + "\n"

                _la = [str(k)] + [
                    KETA[m].format(float(SCORES[task][str(k)][split][m]))
                    for m in METRICS
                ]
                LATEX += _la + ["\\\\ \n"]

        save_path = os.path.join(ROOT, f"scores_{split}.tex")
        with open(save_path, "w") as f:
            LATEX = " & ".join(LATEX)
            LATEX = LATEX.replace("& \\", " \\")
            f.writelines(LATEX)
        print(f"Save: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="results")
    args = parser.parse_args()
    r = args.root
    export_score_as_csv(r)
