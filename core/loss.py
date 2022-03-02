import torch
import torch.nn as nn
import _init_paths


# from configs.configs import *

class LOSS(nn.Module):
    def __init__(self, config):
        super(LOSS, self).__init__()
        self.rotation_weight = config.rotation_weight
        self.trans_weight = config.trans_weight
        self.pos_weight = config.pos_weight
        self.vel_weight = config.vel_weight
        self.gram_weight = config.gram_weight
        self.genre_weight = config.genre_weight
        # self.cal_len = config.cal_len

    def _calc_loss_(self, output, targets, genre_label=None, total=True):
        min_len = min(output[0].shape[1], targets.shape[1])
        targets = targets[:, :min_len, :]
        output[2] = output[2][:, :min_len, :]
        rotation_pose9d = output[0][:, :, :216]
        rotation_trans = output[0][:, :, 216:219]

        targets_pose9d = targets[:, :, :216]
        targets_trans = targets[:, :, 216:219]

        rotation_loss = _calc_joint_rotation_loss_(rotation_pose9d, targets_pose9d)
        trans_loss = _calc_joint_trans_loss_(rotation_trans, targets_trans)

        total_loss = self.rotation_weight * rotation_loss + self.trans_weight * trans_loss

        if total == True:
            joint_position_loss = _calc_joint_position_loss_(output[1], output[2])
            joint_velocity_loss = _calc_joint_velocity_loss_(output[1], output[2])
            gram_matrix_loss = _calc_gram_matrix_loss_2(output[0], targets)
            if genre_label is not None:
                genres_loss = _calc_genres_loss_(output[3], genre_label)
                total_loss += self.pos_weight * joint_position_loss + self.vel_weight * joint_velocity_loss \
                              + self.gram_weight * gram_matrix_loss + self.genre_weight * genres_loss
            else:
                genres_loss = torch.Tensor([0.])
                total_loss += self.pos_weight * joint_position_loss + self.vel_weight * joint_velocity_loss \
                              + self.gram_weight * gram_matrix_loss
            return (total_loss, self.rotation_weight * rotation_loss, self.trans_weight * trans_loss, \
                    self.pos_weight * joint_position_loss, self.vel_weight * joint_velocity_loss, \
                    self.gram_weight * gram_matrix_loss, self.genre_weight * genres_loss)

        else:
            return (total_loss, self.rotation_weight * rotation_loss, self.trans_weight * trans_loss)

    def _calc_all_loss_(self, output, targets):
        return (nn.MSELoss(reduce=True, size_average=True)(output, targets))


def _calc_joint_rotation_loss_(preds, targets):  # (batch size)
    # return torch.mean(torch.norm((targets-preds), p=2, dim=-1).mean(-1).mean(-1).mean(-1))
    return nn.MSELoss(reduce=True, size_average=True)(preds, targets)


def _calc_joint_trans_loss_(preds, targets):  # (batch size)
    # return torch.mean(torch.norm(targets-preds, p=1, dim=-1).mean(-1))
    # return torch.mean(torch.norm((targets-preds), p=2, dim=-1).mean(-1).mean(-1))
    return nn.MSELoss(reduce=True, size_average=True)(preds, targets)


def _calc_joint_all_loss_(preds, targets):  # (batch size)
    # return torch.mean(torch.norm(targets-preds, p=1, dim=-1).mean(-1))
    # return torch.mean(torch.norm((targets-preds), p=2, dim=-1).mean(-1).mean(-1))
    return nn.MSELoss(reduce=True, size_average=True)(preds, targets)


def _calc_gram_matrix_loss_(preds, targets):  # (batch size, seq_len, 219)
    gram_pred = torch.bmm(preds, preds.transpose(1, 2))
    gram_tgt = torch.bmm(targets, targets.transpose(1, 2))
    return nn.MSELoss(reduce=True, size_average=True)(gram_pred, gram_tgt)


def _calc_gram_matrix_loss_2(preds, targets):  # (batch size, seq_len, 219)
    preds_tmp = torch.cat([preds[:-1], preds[1:]], dim=-1)
    targets_tmp = torch.cat([targets[:-1], targets[1:]], dim=-1)
    gram_pred = torch.bmm(preds_tmp, preds_tmp.transpose(1, 2))
    gram_tgt = torch.bmm(targets_tmp, targets_tmp.transpose(1, 2))
    return nn.MSELoss(reduce=True, size_average=True)(gram_pred, gram_tgt)


def _calc_genres_loss_(preds, targets):  # (batch size)
    return nn.CrossEntropyLoss()(preds, targets)


def _calc_joint_position_loss_(pred_joints, target_joints):
    # joint_position_loss = torch.mean(torch.norm((pred_joints-target_joints), p=2, dim=-1).mean(-1))
    joint_position_loss = nn.MSELoss(reduce=True, size_average=True)(pred_joints, target_joints)

    return joint_position_loss


def _calc_joint_velocity_loss_(pred_joints, target_joints):
    bs, seq_len, _, _ = pred_joints.shape
    pred_joints = pred_joints.reshape(bs, seq_len, -1)
    target_joints = target_joints.reshape(bs, seq_len, -1)
    velocity_predicted = pred_joints[:, 1:] - pred_joints[:, :-1]
    velocity_target = target_joints[:, 1:] - target_joints[:, :-1]
    # velocity_predicted = torch.diff(pred_joints, dim=0)
    # velocity_target = torch.diff(target_joints, dim=0)
    # joint_velocity_loss = torch.mean(torch.norm((velocity_predicted-velocity_target), p=2, dim=-1).mean(-1))
    joint_velocity_loss = nn.MSELoss(reduce=True, size_average=True)(velocity_predicted, velocity_target)

    return joint_velocity_loss
