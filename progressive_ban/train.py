
import os
import random
import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import wandb
from tqdm import trange

from pytorchcv.model_provider import get_model as ptcv_get_model


# ─── Helpers for layer freezing ───────────────────────────────────────────────

def get_layer_groups(model: nn.Module):
    """
    Split a CIFARResNet (from pytorchcv) into:
      0: init_block, 1: stage1, 2: stage2, 3: stage3, 4: output head
    """
    feats = model.features
    return [
        feats.init_block,
        feats.stage1,
        feats.stage2,
        feats.stage3,
        model.output,
    ]

def copy_and_freeze_layers(student: nn.Module,
                           teacher: nn.Module,
                           up_to_stage: int):
    """
    Copies params & buffers from teacher→student for layer-groups [0..up_to_stage],
    then freezes those params so they won't be updated (and locks BN stats).
    """
    s_groups = get_layer_groups(student)
    t_groups = get_layer_groups(teacher)
    assert 0 <= up_to_stage < len(s_groups), \
        f"up_to_stage must be in [0..{len(s_groups)-1}]"

    for idx in range(up_to_stage + 1):
        s_blk = s_groups[idx]
        t_blk = t_groups[idx]

        # 1) Copy weights & running stats
        s_blk.load_state_dict(t_blk.state_dict(), strict=True)

        # 2) Freeze parameters and put BN into eval mode
        for p in s_blk.parameters():
            p.requires_grad = False
        for m in s_blk.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()


# ─── Setup + seed ─────────────────────────────────────────────────────────────

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"🖥️  Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

seed = int(os.environ.get("SEED", 42))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

rand_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

config = dict(
    seed         = seed,
    dataset      = "CIFAR-100",
    teacher_arch = "ResNet56",
    student_arch = "ResNet56",
    batch_size   = 128,
    lr           = 0.1,
    weight_decay = 1e-4,
    ban_gens     = 3,     # how many KD generations
    ban_alpha    = 1.0,
    ban_T        = 4.0,
    ban_epochs   = 10,
)

# ─── Data ─────────────────────────────────────────────────────────────────────

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

train_ds = CIFAR100(DATA_DIR, train=True,  download=True, transform=train_transform)
test_ds  = CIFAR100(DATA_DIR, train=False, download=True, transform=test_transform)

g = torch.Generator().manual_seed(seed)
train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                          num_workers=4, pin_memory=True,
                          worker_init_fn=seed_worker, generator=g)
test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'], shuffle=False,
                          num_workers=4, pin_memory=True)

# ─── Models & initial eval ───────────────────────────────────────────────────

teacher_name = config["teacher_arch"] + "_cifar100"
student_name = config["student_arch"] + "_cifar100"

teacher_model = ptcv_get_model(teacher_name, pretrained=True).to(device)
# put teacher into eval mode once (locks dropout & BN stats)
prev_teacher = teacher_model.eval()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct/total

print(f"Teacher acc: {evaluate(prev_teacher, test_loader)*100:.2f}%")

# ─── WandB init for overall best record ───────────────────────────────────────

wandb.init(project="LoRA-KD", name=f"overall_master_{rand_id}",
           config=config, reinit=True)
wandb.log({"teacher/test_acc": evaluate(prev_teacher, test_loader)})
wandb.finish()

# ─── KD loss ──────────────────────────────────────────────────────────────────

def kd_loss_fn(s_logits, t_logits, targets,
               T=config['ban_T'], alpha=config['ban_alpha']):
    ce = F.cross_entropy(s_logits, targets)
    kd = F.kl_div(
        F.log_softmax(s_logits/T, dim=1),
        F.softmax(t_logits/T, dim=1),
        reduction='batchmean'
    ) * (T*T)
    return alpha*kd + (1-alpha)*ce

# ─── Successive generations ──────────────────────────────────────────────────

overall_best_acc   = 0.0
overall_best_state = None
overall_best_gen   = None

for gen in trange(1, config['ban_gens']+1, desc="BAN Generations"):
    run = wandb.init(
        project="LoRA-KD",
        name=f"gen_{gen}_{rand_id}",
        group=rand_id,
        config=config,
        reinit=True
    )
    # fresh student each gen
    student = ptcv_get_model(student_name, pretrained=False).to(device)
    wandb.watch(student, log="all", log_freq=100)

    # freeze progressively deeper layers if gen>1
    if gen > 1:
        copy_and_freeze_layers(student, prev_teacher, up_to_stage=gen)

    opt   = optim.SGD(student.parameters(),
                      lr=config['lr'],
                      momentum=0.9,
                      weight_decay=config['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt,
                                                config['ban_epochs'])

    best_acc_epoch = 0.0
    best_state     = None

    for epoch in trange(1, config['ban_epochs']+1,
                        desc=f"Gen {gen} Ep", leave=False):
        # — train —
        student.train()
        running_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.no_grad():
                t_logits = prev_teacher(x)
            s_logits = student(x)
            loss = kd_loss_fn(s_logits, t_logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item()*y.size(0)
            correct      += (s_logits.argmax(1)==y).sum().item()
            total        += y.size(0)

        train_loss = running_loss/total
        train_acc  = correct/total

        # — validate —
        student.eval()
        running_loss = correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                t_logits = prev_teacher(x)
                s_logits = student(x)
                loss = kd_loss_fn(s_logits, t_logits, y)
                running_loss += loss.item()*y.size(0)
                correct      += (s_logits.argmax(1)==y).sum().item()
                total        += y.size(0)

        val_loss = running_loss/total
        val_acc  = correct/total

        sched.step()
        wandb.log({
            "train/loss": train_loss,
            "train/acc":  train_acc,
            "val/loss":   val_loss,
            "val/acc":    val_acc,
            "lr":         opt.param_groups[0]['lr'],
            "epoch":      epoch,
        }, step=epoch)

        if val_acc > best_acc_epoch:
            best_acc_epoch = val_acc
            best_state     = student.state_dict()

    # save & log
    model_path = f"best_model_gen{gen}.pt"
    torch.save(best_state, model_path)
    art = wandb.Artifact(name=f"gen_{gen}", type="best_gen_model")
    art.add_file(model_path)
    run.log_artifact(art)
    print(f"✅ Gen {gen} best val-acc: {best_acc_epoch*100:.2f}%")

    # update overall best
    if best_acc_epoch > overall_best_acc:
        overall_best_acc   = best_acc_epoch
        overall_best_state = best_state
        overall_best_gen   = gen

    # promote this student → next teacher
    prev_teacher.load_state_dict(best_state)
    prev_teacher.eval()
    wandb.finish()

# ─── Log overall best ────────────────────────────────────────────────────────

final_run = wandb.init(
    project="LoRA-KD",
    name=f"overall_best_{rand_id}",
    group=rand_id,
    reinit=True
)
torch.save(overall_best_state, "best_model_overall.pt")
overall_art = wandb.Artifact("best_model_overall", type="best_model")
overall_art.add_file("best_model_overall.pt")
final_run.log_artifact(overall_art)
final_run.finish()

print(f"🏆 Finished all gens. Overall best = Gen {overall_best_gen} @ {overall_best_acc*100:.2f}%")

