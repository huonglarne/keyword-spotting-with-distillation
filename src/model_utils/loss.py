from torch import nn

def _student_loss(student_preds, labels):
    loss = nn.CrossEntropyLoss()
    return loss(student_preds, labels)

def _kd_loss(student_preds, teacher_preds, T):
    loss = nn.KLDivLoss(
        reduction="batchmean",
        log_target=True
    )

    return loss(
        (student_preds/T).float(),
        (teacher_preds/T).float(),
    ) * T * T

def distillation_loss(student_preds, teacher_preds, labels, alpha=0.1, T=10):
    student_loss = _student_loss(student_preds, labels)
    kd_loss = _kd_loss(student_preds, teacher_preds, T)

    return kd_loss * alpha + student_loss * (1 - alpha)