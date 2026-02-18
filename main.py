import numpy as np
import torch
import os
import src.config as cfg
from src.data_loader import load_all_subjects_data
from src.dataset import build_loaders_for_loso
from src.train import train_one_fold
from src.analysis import get_channel_link_matrix, plot_channel_link_matrix, plot_cv_accuracy_over_epochs

def main():
    # 1. Load all data
    print("Loading data...")
    per_subject_data = load_all_subjects_data()
    available_subjects = sorted(per_subject_data.keys())
    
    if not available_subjects:
        print("No data found. Check paths in src/config.py")
        return

    # 2. Storage for results
    fold_results = {}
    fold_histories = {}
    
    # 3. LOSO Cross-Validation Loop
    for test_subj in available_subjects:
        print(f"\n=== LOSO Fold: Testing on Subject {test_subj} ===")
        
        # Prepare DataLoaders
        train_loader, test_loader, brain_regions = build_loaders_for_loso(
            per_subject_data, test_subj, batch_size=cfg.BATCH_SIZE
        )
        
        # Train Model
        model, history, confusionMatrix = train_one_fold(
            train_loader, test_loader, num_classes=cfg.NUM_CLASSES, 
            num_epochs=cfg.NUM_EPOCHS, lr=cfg.LEARNING_RATE
        )
        
        # Store results
        fold_results[test_subj] = history["test_acc"][-1]
        fold_histories[test_subj] = history
        
        # Analysis: Channel Link Matrix
        chan_corr = get_channel_link_matrix(model)
        save_path = os.path.join(cfg.RESULTS_DIR, f"Subject_{test_subj}_Heatmap.png")
        plot_channel_link_matrix(chan_corr, brain_regions, 
                                 title=f"Subj {test_subj} Connectivity", 
                                 save_path=save_path)

    # 4. Summary
    print("\nPer-subject LOSO accuracies:")
    for subj_id, acc in fold_results.items():
        print(f"  {subj_id}: {acc:.3f}")
    
    print(f"\nMean LOSO accuracy: {np.mean(list(fold_results.values())):.3f}")
    
    # Plot aggregate accuracy
    plot_cv_accuracy_over_epochs(fold_histories, metric="test_acc", 
                                 save_path=os.path.join(cfg.RESULTS_DIR, "mean_accuracy.png"))

if __name__ == "__main__":
    main()