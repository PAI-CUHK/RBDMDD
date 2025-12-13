def plot_confusion_matrices(
    items: Sequence[Tuple[str, np.ndarray]], output_path: Path
) -> None:
    if not items:
        raise SystemExit("No confusion matrices to plot.")

    num_plots = len(items)
    
   
    cols = min(num_plots, 3) 
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    
    if num_plots == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(rows, cols)

   
    vmax = max(matrix.max() for _, matrix in items)
    
    for idx, (title, matrix) in enumerate(items):
        ax = axes.flatten()[idx]
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax if vmax else 1)
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                color = "white" if value > (vmax * 0.6 if vmax else 0) else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=11)
        
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS)
        ax.set_yticks(range(len(LABELS)))
        ax.set_yticklabels(LABELS)
        
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        ax.grid(False)

    for ax in axes.flatten()[num_plots:]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
