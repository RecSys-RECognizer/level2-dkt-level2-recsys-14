import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.nn.models import LightGCN
from .scheduler import get_scheduler



def build_tag(n_node, weight=None, logger=None, **kwargs):
    model = LightGCN(n_node, **kwargs)
    if weight:
        if not os.path.isfile(weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def train(
    model,
    train_data,
    valid_data=None,
    n_epoch=100,
    learning_rate=0.01,
    use_wandb=False,
    weight=None,
    logger=None,
):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_scheduler(optimizer)

    if not os.path.exists(weight):
        os.makedirs(weight)

    if valid_data is None:
        print("valid_data is None!!")
        #eids = np.arange(len(train_data["label"]))
        #eids = np.random.permutation(eids)[:1000]
        #edge, label = train_data["edge"], train_data["label"]
        #label = label.to("cpu").detach().numpy()
        #valid_data = dict(edge=edge[:, eids], label=label[eids])

        ##new method
        valid_ratio = 0.3
        torch.manual_seed(42)
        perm = torch.randperm(len(train_data['label']))
        valid_new_size = int(len(train_data['label'])*valid_ratio)
        print('train data size: ', len(train_data['label']))
        print('valid data size: ', valid_new_size)

        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge, label=label)

        valid_new_edges = perm[:valid_new_size]
        train_data['label'][valid_new_edges] = 0


    else:
        print("valid_data is already set!!")
        edge, label = valid_data["edge"], valid_data["label"]
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge, label=label)


    logger.info(f"Training Started : n_epoch={n_epoch}")
    best_auc, best_epoch = 0, -1
    best_loss = 10
    best_acc = 0
    for e in range(n_epoch):
        # forward
        pred = model(train_data["edge"])
        loss = model.link_pred_loss(pred, train_data["label"])

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #update_params(loss, model, optimizer, scheduler)

        with torch.no_grad():
            prob = model.predict_link(valid_data["edge"], prob=True)
            prob = prob.detach().cpu().numpy()
            acc = accuracy_score(valid_data["label"], prob > 0.5)
            auc = roc_auc_score(valid_data["label"], prob)
            logger.info(
                f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}"
            )
            if use_wandb:
                import wandb

                wandb.log(dict(loss=loss, acc=acc, auc=auc))

        if weight:
            if auc > best_auc:
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Best AUC"
                )
                best_auc, best_epoch = auc, e
                torch.save(
                    {"model": model.state_dict(), "epoch": e + 1},
                    os.path.join(weight, f"best_auc_model_tag.pt"),
                )
            if best_loss > loss:
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Best LOSS"
                )
                best_loss, best_epoch = loss, e
                torch.save(
                    {"model": model.state_dict(), "epoch": e + 1},
                    os.path.join(weight, f"best_loss_model.pt"),
                )
            if acc > best_acc:
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Best ACC"
                )
                best_acc, best_epoch = acc, e
                torch.save(
                    {"model": model.state_dict(), "epoch": e + 1},
                    os.path.join(weight, f"best_acc_model.pt"),
                )
    torch.save(
        {"model": model.state_dict(), "epoch": e + 1},
        os.path.join(weight, f"last_model.pt"),
    )
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")

def update_params(loss, model, optimizer, scheduler):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10) #clip_grad=10
    # if args.scheduler == "linear_warmup":
    #    scheduler.step()
    scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def inference(model, data, logger=None):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(data["edge"], prob=True)
        return pred
