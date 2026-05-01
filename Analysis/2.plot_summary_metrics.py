def build_fold_metrics(fold_subjects: Dict[str, Dict[str, Dict[str, List[float]]]]) -> List[FoldMetrics]:
    metrics: List[FoldMetrics] = []
    for fold, subjects in fold_subjects.items():
        y_true: List[int] = []
        y_score: List[float] = []
        for subject in sorted(subjects):
            info = subjects[subject]
            y_true.append(info["true_label"])
            y_score.append(float(np.mean(info["scores"])) if info["scores"] else 0.0)
        y_pred = [int(score >= SUBJECT_THRESHOLD) for score in y_score]
        values = _compute_metrics_from_preds(y_true, y_pred)
        metrics.append(
            FoldMetrics(
                fold_label=fold,
                subjects=len(y_true),
                accuracy=values["accuracy"],
                balanced_accuracy=values["balanced_accuracy"],
                precision=values["precision"],
                recall=values["recall"],
                specificity=values["specificity"],
                f1=values["f1"],
            )
        )
    metrics.sort(key=lambda item: item.fold_label)
    return metrics


def plot_metrics(metrics: Sequence[FoldMetrics], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1"]
    metric_labels = ["Accuracy", "Bal. Acc", "Precision", "Recall", "Specificity", "F1"]
    colors = [
        "#C6C5E3",  
        "#6FA3D3",  
        "#EB8673",  
        "#95C78E",  
        "#F0CF7F",  
        "#D99BBD",  
    ]

    ax_left = axes[0]
    x = np.arange(len(metrics))
    width = 0.13
    for offset, (metric_name, label, color) in enumerate(zip(metric_names, metric_labels, colors)):
        values = [getattr(m, metric_name) * 100 for m in metrics]
        ax_left.bar(x + offset * width, values, width, label=label, color=color, alpha=0.85)
    ax_left.set_xlabel("Fold")
    ax_left.set_ylabel("Value (%)")
    ax_left.set_title("Patient-level Metrics per Fold", fontweight="bold")
    ax_left.set_xticks(x + width * 2.5)
    ax_left.set_xticklabels([m.fold_label for m in metrics])
    ax_left.set_ylim(0, 105)
    ax_left.legend(loc="lower right", fontsize=9)
    ax_left.grid(True, axis="y", alpha=0.3)

    ax_right = axes[1]
    avg_values = []
    std_values = []
    for metric_name in metric_names:
        values = np.array([getattr(m, metric_name) for m in metrics])
        avg_values.append(values.mean() * 100)
        std_values.append(values.std(ddof=0) * 100)
    x_pos = np.arange(len(metric_labels))
    bars = ax_right.bar(x_pos, avg_values, yerr=std_values, capsize=5, color=colors, alpha=0.85)
    for bar, avg, std in zip(bars, avg_values, std_values):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 1,
            f"{avg:.1f}±{std:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax_right.set_xlabel("Metric")
    ax_right.set_ylabel("Value (%)")
    ax_right.set_title("Patient-level Mean ± Std", fontweight="bold")
    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(metric_labels)
    ax_right.set_ylim(0, 105)
    ax_right.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Metrics Comparison (patient-level)", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
