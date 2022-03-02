"""
SPL: training and evaluation of neural networks with a structured prediction layer.
Copyright (C) 2019 ETH Zurich, Emre Aksan, Manuel Kaufmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import cv2

SMPL_NR_JOINTS = 24
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPL_JOINTS = [
    'pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle', 'spine3',
    'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
    'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand']
SMPL_JOINT_MAPPING = {i: x for i, x in enumerate(SMPL_JOINTS)}


class SMPLForwardKinematics(object):
    """
    FK Engine.
    """
    def __init__(self):
        # this are the offsets stored under `J` in the SMPL model pickle file
        offsets = np.array([[-8.76308970e-04, -2.11418723e-01, 2.78211200e-02],
                            [7.04848876e-02, -3.01002533e-01, 1.97749280e-02],
                            [-6.98883278e-02, -3.00379160e-01, 2.30254335e-02],
                            [-3.38451650e-03, -1.08161861e-01, 5.63597909e-03],
                            [1.01153808e-01, -6.65211904e-01, 1.30860155e-02],
                            [-1.06040718e-01, -6.71029623e-01, 1.38401121e-02],
                            [1.96440985e-04, 1.94957852e-02, 3.92296547e-03],
                            [8.95999143e-02, -1.04856032e+00, -3.04155922e-02],
                            [-9.20120818e-02, -1.05466743e+00, -2.80514913e-02],
                            [2.22362284e-03, 6.85680141e-02, 3.17901760e-02],
                            [1.12937580e-01, -1.10320516e+00, 8.39545265e-02],
                            [-1.14055299e-01, -1.10107698e+00, 8.98482216e-02],
                            [2.60992373e-04, 2.76811197e-01, -1.79753042e-02],
                            [7.75218998e-02, 1.86348444e-01, -5.08464100e-03],
                            [-7.48091986e-02, 1.84174211e-01, -1.00204779e-02],
                            [3.77815350e-03, 3.39133394e-01, 3.22299558e-02],
                            [1.62839013e-01, 2.18087461e-01, -1.23774789e-02],
                            [-1.64012068e-01, 2.16959041e-01, -1.98226746e-02],
                            [4.14086325e-01, 2.06120683e-01, -3.98959248e-02],
                            [-4.10001734e-01, 2.03806676e-01, -3.99843890e-02],
                            [6.52105424e-01, 2.15127546e-01, -3.98521818e-02],
                            [-6.55178550e-01, 2.12428626e-01, -4.35159074e-02],
                            [7.31773168e-01, 2.05445019e-01, -5.30577698e-02],
                            [-7.35578759e-01, 2.05180646e-01, -5.39352281e-02]])

        # need to convert them to compatible offsets
        smpl_offsets = np.zeros([24, 3])
        smpl_offsets[0] = offsets[0]
        for idx, pid in enumerate(SMPL_PARENTS[1:]):
            smpl_offsets[idx+1] = offsets[idx + 1] - offsets[pid]
        
        self.offsets = smpl_offsets
        self.parents = SMPL_PARENTS
        self.n_joints = SMPL_NR_JOINTS
        assert self.offsets.shape[0] == self.n_joints

    def fk(self, joint_angles, joint_trans=None):
        """
        Perform forward kinematics. This requires joint angles to be in rotation matrix format.
        Args:
            joint_angles: np array of shape (N, n_joints*3*3)
            joint_trans: np array of shape (N, 3)

        Returns:
            The 3D joint positions as a an array of shape (N, n_joints, 3)
        """
        assert joint_angles.shape[-1] == self.n_joints * 9
        angles = np.reshape(joint_angles, [-1, self.n_joints, 3, 3])
        n_frames = angles.shape[0]
        positions = np.zeros([n_frames, self.n_joints, 3])
        rotations = np.zeros([n_frames, self.n_joints, 3, 3])  # intermediate storage of global rotation matrices
        offsets = self.offsets[np.newaxis, ..., np.newaxis]  # (1, n_joints, 3, 1)

        for j in range(self.n_joints):
            if self.parents[j] == -1:
                # this is the root, we don't consider any root translation
                positions[:, j] = 0.0
                rotations[:, j] = angles[:, j]
            else:
                # this is a regular joint
                positions[:, j] = np.squeeze(np.matmul(rotations[:, self.parents[j]], offsets[:, j])) + \
                                    positions[:, self.parents[j]]
                rotations[:, j] = np.matmul(rotations[:, self.parents[j]], angles[:, j])

        # apply global scaling and translation:
        # be careful of the order here: first apply the scaling, then the translation.
        if joint_trans is not None:
            positions += joint_trans.reshape(n_frames, 1, 3)
        
        return positions

    def from_aa(self, joint_angles, joint_trans=None):
        """
        Get joint positions from angle axis representations in shape (N, n_joints*3).
        """
        angles = np.reshape(joint_angles, [-1, self.n_joints, 3])
        angles_rot = np.zeros(angles.shape + (3,))
        for i in range(angles.shape[0]):
            for j in range(self.n_joints):
                angles_rot[i, j] = cv2.Rodrigues(angles[i, j])[0]
        return self.fk(np.reshape(angles_rot, [-1, self.n_joints * 9]), joint_trans)

    def from_rotmat(self, joint_angles, joint_trans=None):
        """
        Get joint positions from rotation matrix representations in shape (N, H36M_NR_JOINTS*3*3).
        """
        return self.fk(joint_angles, joint_trans)