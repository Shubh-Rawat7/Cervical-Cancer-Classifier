from pathlib import Path

path = Path('notebooks/Train_Cervical_Kaggle_v17.ipynb')
text = path.read_text(encoding='utf-8')
old = '''OPTIONAL_FLAGS = {
    '--img-size':           str(IMAGE_SIZE),
    '--val-split':          str(VAL_SPLIT),
    '--lr':                 str(LR),
    '--lr-head':            str(LR),
    '--lr-backbone':        str(max(LR / 10.0, 1e-7)),
    '--weight-decay':       str(WEIGHT_DECAY),
    '--patience':           str(PATIENCE),
    '--workers':            str(WORKERS),
    '--seed':               str(SEED),
    '--img-size':           str(IMAGE_SIZE),
    '--val-split':          str(VAL_SPLIT),
    '--lr':                 str(LR),
    '--lr-head':            str(LR),
    '--lr-backbone':        str(max(LR / 10.0, 1e-7)),
    '--weight-decay':       str(WEIGHT_DECAY),
    '--workers':            str(WORKERS),
    '--seed':               str(SEED),
    '--img-size':           str(IMAGE_SIZE),
    '--val-split':          str(VAL_SPLIT),
    '--lr':                 str(LR),
    '--lr-head':            str(LR),
    '--lr-backbone':        str(max(LR / 10.0, 1e-7)),
    '--weight-decay':       str(WEIGHT_DECAY),
    '--workers':            str(WORKERS),
    '--seed':               str(SEED),
    '--img-size':           str(IMAGE_SIZE),
    '--val-split':          str(VAL_SPLIT),
    '--lr':                 str(LR),
    '--lr-head':            str(LR),
    '--lr-backbone':        str(max(LR / 10.0, 1e-7)),
    '--weight-decay':       str(WEIGHT_DECAY),
    '--workers':            str(WORKERS),
    '--seed':               str(SEED),
    '--accumulation-steps': str(ACCUMULATION_STEPS),
    '--loss-type':          LOSS_TYPE,
    '--undersample':        UNDERSAMPLE,
    '--gamma':              str(GAMMA),
    '--beta':               str(BETA),
    '--label-smoothing':    str(LABEL_SMOOTHING),
    '--activation':         ACTIVATION,
    '--dropout':            str(DROPOUT),
    '--backbone':           BACKBONE,
    '--warmup-epochs':      str(int(os.environ.get('WARMUP_EPOCHS', '10'))),
    '--mixup-alpha':        str(MIXUP_ALPHA),
    '--cutmix-alpha':       str(CUTMIX_ALPHA),
}
'''
new = '''OPTIONAL_FLAGS = {
    '--img-size':           str(IMAGE_SIZE),
    '--val-split':          str(VAL_SPLIT),
    '--lr-head':            str(LR),
    '--lr-backbone':        str(max(LR / 10.0, 1e-7)),
    '--weight-decay':       str(WEIGHT_DECAY),
    '--patience':           str(PATIENCE),
    '--workers':            str(WORKERS),
    '--seed':               str(SEED),
    '--accumulation-steps': str(ACCUMULATION_STEPS),
    '--loss-type':          LOSS_TYPE,
    '--undersample':        UNDERSAMPLE,
    '--gamma':              str(GAMMA),
    '--beta':               str(BETA),
    '--label-smoothing':    str(LABEL_SMOOTHING),
    '--activation':         ACTIVATION,
    '--dropout':            str(DROPOUT),
    '--backbone':           BACKBONE,
    '--warmup-epochs':      str(int(os.environ.get('WARMUP_EPOCHS', '10'))),
    '--mixup-alpha':        str(MIXUP_ALPHA),
    '--cutmix-alpha':       str(CUTMIX_ALPHA),
}
'''
old2 = '''# Ensure LR/epochs/batch explicitly passed only if supported by train.py
for flag, value in [('--lr', str(LR)), ('--epochs', str(EPOCHS)), ('--batch-size', str(BATCH_SIZE))]:
    if flag in help_text:
        extra_flags += [flag, value]

cmd.extend(extra_flags)
'''
new2 = '''# Ensure custom experiment flags are appended only if train.py supports them.
# --epochs and --batch-size are already included in the required base command.
cmd.extend(extra_flags)
'''
if old not in text:
    raise RuntimeError('Old OPTIONAL_FLAGS block not found')
text = text.replace(old, new)
if old2 not in text:
    raise RuntimeError('Old extra_flags loop not found')
text = text.replace(old2, new2)
path.write_text(text, encoding='utf-8')
print('patched')
