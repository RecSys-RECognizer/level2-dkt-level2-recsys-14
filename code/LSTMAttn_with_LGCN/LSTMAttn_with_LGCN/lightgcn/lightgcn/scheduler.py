from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup


def get_scheduler(optimizer):
    #if args.scheduler == "plateau":
    #    scheduler = ReduceLROnPlateau(
    #        optimizer, patience=10, factor=0.5, mode="max", verbose=True
    #    )
    #elif args.scheduler == "linear_warmup":
    #    scheduler = get_linear_schedule_with_warmup(
    #        optimizer,
    #        num_warmup_steps=args.warmup_steps,
    #        num_training_steps=args.total_steps,
    #    )
    ## "linear_warmup":
    #scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode="max", verbose=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    return scheduler
