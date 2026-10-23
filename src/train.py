import random
import copy
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def set_seed(seed: int):
    """Set random seeds across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_device() -> torch.device:
    """Detect and return optimal device: MPS if available, else CUDA, else CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def compute_pair_accuracy(ids: np.ndarray, y: np.ndarray, scores: np.ndarray) -> float:
    """
    Computes pair accuracy: for each caption id (pristine/falsified pair),
    the fraction of pairs where the model gives a higher falsified score
    to the falsified sample than to the pristine one.
    """
    N = len(ids)
    if N % 2 == 0 and np.all(ids[0::2] == ids[1::2]) and np.all(y[0::2] == 0) and np.all(y[1::2] == 1):
        pristine_scores = scores[0::2]
        falsified_scores = scores[1::2]
        return float(np.mean(falsified_scores > pristine_scores))
    
    # Fallback group by caption id
    from collections import defaultdict
    by_id = defaultdict(dict)
    for cid, label, score in zip(ids, y, scores):
        by_id[cid][label] = score
        
    correct = 0
    total = 0
    for cid, d in by_id.items():
        if 0 in d and 1 in d:
            total += 1
            if d[1] > d[0]:
                correct += 1
    return float(correct / total) if total > 0 else 0.0

def evaluate_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    ids: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics given true binary labels and continuous falsified scores.
    """
    preds = (scores >= threshold).astype(int)
    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, scores))
    cm = confusion_matrix(y_true, preds).tolist()
    pair_acc = compute_pair_accuracy(ids, y_true, scores)
    
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pair_accuracy": pair_acc,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float,
    batch_size: int,
    epochs_max: int,
    early_stopping_patience: int,
    seed: int,
    device: torch.device
) -> Tuple[nn.Module, Dict[str, List[float]], int, float]:
    """
    Train model using Adam, BCEWithLogitsLoss, and early stopping on validation loss.
    Restores the best weights before returning.
    """
    set_seed(seed)
    model = model.to(device)
    
    # DataLoader setup
    generator = torch.Generator().manual_seed(seed)
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(1, epochs_max + 1):
        # Training phase
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item() * batch_x.size(0)
            train_samples += batch_x.size(0)
            
        train_epoch_loss = train_loss_total / train_samples
        history["train_loss"].append(train_epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                val_loss_total += loss.item() * batch_x.size(0)
                val_samples += batch_x.size(0)
                
        val_epoch_loss = val_loss_total / val_samples
        history["val_loss"].append(val_epoch_loss)
        
        # Check early stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break
                
    # Restore best weights
    model.load_state_dict(best_state_dict)
    return model, history, best_epoch, best_val_loss

def predict_scores(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    """
    Compute sigmoid probabilities (falsified scores) for input features.
    """
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_scores = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            all_scores.append(probs.cpu().numpy())
            
    return np.concatenate(all_scores, axis=0)
