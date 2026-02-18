"""
Training functionality for one fold i.e. one subject.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from src.model import EEGNet
import src.config as cfg

def train_one_fold(train_loader, test_loader, num_classes, num_epochs=20, lr=0.01, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EEGNet(
        num_channels=cfg.NUM_CHANNELS,
        num_samples=cfg.NUM_SAMPLES,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
      "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []
    }

    for epoch in range(num_epochs):
        # Train
        model.train()
        running_train_loss, train_correct, train_total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        avg_train_loss = running_train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Eval
        model.eval()
        running_test_loss, test_correct, test_total = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                running_test_loss += loss.item() * yb.size(0)
                preds = logits.argmax(dim=1)
                test_correct += (preds == yb).sum().item()
                test_total += yb.size(0)

        avg_test_loss = running_test_loss / max(test_total, 1)
        test_acc = test_correct / max(test_total, 1)

        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(avg_test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1:02d} | train_loss: {avg_train_loss:.4f}, train_acc: {train_acc:.3f} | test_loss: {avg_test_loss:.4f}, test_acc: {test_acc:.3f}")

    # Final confusion matrix
    all_preds, all_true = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())

    cm = None
    if len(all_true) > 0:
        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        cm = confusion_matrix(all_true, all_preds)
        print("Confusion Matrix:\n", cm)

    return model, history, cm