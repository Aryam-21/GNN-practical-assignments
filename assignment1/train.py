"""Training loop: loss -> backprop -> weight update; keeps best-validation checkpoint."""
import torch
import torch.nn.functional as F


def accuracy(logits, y, mask):
    return (logits[mask].argmax(1) == y[mask]).float().mean().item()


def train(model, x, edge_index, y, masks, epochs=200, lr=0.01, wd=5e-4):
    tr, va, te = masks
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    hist = {"loss": [], "train": [], "val": []}
    best_val, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, edge_index)[tr], y[tr])   # loss on TRAIN labels only
        loss.backward()                                            # backprop
        opt.step()                                                 # weight update

        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        hist["loss"].append(loss.item())
        hist["train"].append(accuracy(out, y, tr))
        hist["val"].append(accuracy(out, y, va))
        if hist["val"][-1] > best_val:                             # model selection on VAL
            best_val = hist["val"][-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_acc = accuracy(model(x, edge_index), y, te)           # TEST used once, at the end
    return hist, best_val, test_acc