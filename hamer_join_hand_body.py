import argparse
import numpy as np
import os
import pickle

from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--body", type=str, default="none", required=False)
parser.add_argument("--smpl_pkl_path", type=str, required=False)
parser.add_argument("--personid", type=int, default= 0,required=False) # is is used with phmr
# parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()



### Compute global rotation function
def compute_global_rotation(pose_axis_anges, joint_idx):
    """
    calculating joints' global rotation
    Args:
        pose_axis_anges (np.array): SMPLX's local pose (22,3)
    Returns:
        np.array: (3, 3)
    """
    global_rotation = np.eye(3)
    parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19]
    while joint_idx != -1:
        joint_rotation = R.from_rotvec(pose_axis_anges[joint_idx]).as_matrix()
        global_rotation = joint_rotation @ global_rotation
        joint_idx = parents[joint_idx]
    return global_rotation

M = np.diag([-1, 1, 1])    



file = args.input
with(open(file, "rb")) as f:
    pkl_hamer = pickle.load(f)


pkl_out = os.path.join(os.path.dirname(file),"_BodyNone_Hammer_Final_com_mao.pkl")



####################
## Body selection
####################
if args.body == "none":
    smplx_params = {}
    smplx_params["global_orient"] = np.zeros((qtd,3))
    smplx_params["body_pose"] = np.zeros((qtd,63))
    smplx_params["transl"] = np.zeros((qtd,3))

if args.body == "gvhmr":
    with(open(args.smpl_pkl_path, "rb")) as f:
        pkl_gvhmr = pickle.load(f)

    smplx_params = {}
    smplx_params["global_orient"] = pkl_gvhmr["smpl_params_global"]["global_orient"]
    smplx_params["body_pose"] = pkl_gvhmr["smpl_params_global"]["body_pose"]
    smplx_params["transl"] = pkl_gvhmr["smpl_params_global"]["transl"]
    
if args.body == "phmr":
    with(open(args.smpl_pkl_path, "rb")) as f:
        pkl_phmr = pickle.load(f)

    num_phmr_pessoa = args.personid
    smplx_params = {}
    smplx_params["global_orient"] =pkl_phmr[num_phmr_pessoa]["poses"][:,:3] # o [0] é necessario ac que para selecionar a pessoa do phmr
    smplx_params["body_pose"] = pkl_phmr[num_phmr_pessoa]["poses"][:,3:66]
    smplx_params["transl"] = pkl_phmr[num_phmr_pessoa]["trans"]


qtd = min(pkl_hamer['lh_global_orient'].shape[0],smplx_params["global_orient"].shape[0])

# qtd=55
M = np.diag([-1, 1, 1])                                                                                                              # Preparing for the left hand switch

lh_num = 0
rh_num = 0

export_body_hand = {}
export_body_hand["global_orient"] = np.zeros((qtd,3))
export_body_hand["body_pose"] = np.zeros((qtd,63))

export_body_hand["left_hand_pose"] = np.zeros((qtd,45))
export_body_hand["right_hand_pose"] = np.zeros((qtd,45))


# lh_pose_final = np.zeros((45))
# rh_pose_final = np.zeros((45))
body_pose_final = np.zeros((63))




for fra in range(qtd):
    # print('frame: ',fra)
    # Assuming that your data are stored in gvhmr_smplx_params and hamer_mano_params
    # full_body_pose = torch.concatenate((gvhmr_smplx_params["global_orient"][0], gvhmr_smplx_params["body_pose"][0].reshape(21, 3)), dim=0)     # gvhmr_smplx_params["global_orient"]: (3, 3)
    # full_body_pose = np.concatenate((smplx_params["global_orient"][1].reshape(-1,3), smplx_params["body_pose"][1].reshape(21, 3)))     
    full_body_pose = np.concatenate((smplx_params["global_orient"][fra].reshape(-1,3), smplx_params["body_pose"][fra].reshape(-1, 3)))     
    left_elbow_global_rot = compute_global_rotation(full_body_pose, 18) # left elbow IDX: 18
    right_elbow_global_rot = compute_global_rotation(full_body_pose, 19) # left elbow IDX: 19

    # left_wrist_global_rot = hamer_mano_params["global_orient"][0]       #### original apenas para Referencia                                                   # hamer_mano_params["global_orient"]: (2, 3, 3)
    left_wrist_global_rot = pkl_hamer["lh_global_orient"][fra][lh_num][0]                                                                     #lh_global_orient : (frame,  pessoa lh, 1, 3, 3)
    # left_wrist_global_rot = single_pkl_hamer['global_orient'][0]
    left_wrist_global_rot = M @ left_wrist_global_rot @ M                                                                                # mirror switch
    left_wrist_pose = np.linalg.inv(left_elbow_global_rot) @ left_wrist_global_rot

    right_wrist_global_rot = pkl_hamer["rh_global_orient"][fra][rh_num][0]
    # right_wrist_global_rot = single_pkl_hamer['global_orient'][1]
    right_wrist_pose = np.linalg.inv(right_elbow_global_rot) @ right_wrist_global_rot

    last_lw_non_zero_pose = np.eye(3)
    last_rw_non_zero_pose = np.eye(3)

    if (left_wrist_pose == 0).all():
        left_wrist_pose = last_lw_non_zero_pose
    else:
        last_lw_non_zero_pose = left_wrist_pose
    left_wrist_pose_vec = R.from_matrix(left_wrist_pose).as_rotvec()

    if (right_wrist_pose == 0).all():
        right_wrist_pose = last_rw_non_zero_pose
    else:
        last_rw_non_zero_pose = right_wrist_pose
    right_wrist_pose_vec = R.from_matrix(right_wrist_pose).as_rotvec()



    left_hand_pose = np.ones(45)
    right_hand_pose = np.ones(45)
    for i in range(15):
        # left_finger_pose = M @ hamer_mano_params["hand_pose"][0][i] @ M                                                    # hamer_mano_params["hand_pose"]: (2, 15, 3, 3)
        ########## Left fingers
        lh_pose_use = pkl_hamer["lh_pose"][fra][lh_num][i]
        # lh_pose_use = single_pkl_hamer['hand_pose'][0][i] ## estatico para teste
        tmp_lh_pose = np.eye(3)
        if (lh_pose_use == 0).all():
            lh_pose_use = tmp_lh_pose
        else:
            tmp_lh_pose = lh_pose_use

        left_finger_pose = M @ lh_pose_use @ M                                                    # lh_pose: (222, 1, 15, 3, 3) (frames,pessoa lh,15,3,3)
        left_finger_pose_vec = R.from_matrix(left_finger_pose).as_rotvec()
        left_hand_pose[i*3: i*3+3] = left_finger_pose_vec
        
        ########### Right Fingers
        rh_pose_use = pkl_hamer["rh_pose"][fra][rh_num][i]
        # rh_pose_use = single_pkl_hamer['hand_pose'][1][i] ### estatico para teste
        tmp_rh_pose = np.eye(3)
        if (rh_pose_use == 0).all():
            rh_pose_use = tmp_rh_pose
        else:
            tmp_rh_pose = rh_pose_use

        right_finger_pose = rh_pose_use #hamer_mano_params["hand_pose"][1][i]
        right_finger_pose_vec = R.from_matrix(right_finger_pose).as_rotvec()
        right_hand_pose[i*3: i*3+3] = right_finger_pose_vec
    



    body_pose_final[:63] = smplx_params["body_pose"][fra] #fazendo uma copia com um array ja no tamanho esperado esta limi
    # body_pose_final[57: 60] = left_wrist_pose_vec #trocando a direcao do pulso
    # body_pose_final[60: 63] = right_wrist_pose_vec



    # lh_pose_final = left_hand_pose
    # rh_pose_final = right_hand_pose

    # gvhmr_smplx_params["body_pose"][0][57: 60] = left_wrist_pose_vec
    # gvhmr_smplx_params["body_pose"][0][60: 63] = right_wrist_pose_vec
    # gvhmr_smplx_params["left_hand_pose"] = left_hand_pose
    # gvhmr_smplx_params["right_hand_pose"] = right_hand_pose
    export_body_hand["global_orient"][fra] = smplx_params["global_orient"][fra]
    export_body_hand["body_pose"][fra] = body_pose_final
    export_body_hand["left_hand_pose"][fra] = left_hand_pose
    export_body_hand["right_hand_pose"][fra] = right_hand_pose
export_body_hand["transl"] = smplx_params["transl"] 




with(open(pkl_out, "wb")) as f:
    pickle.dump(export_body_hand, f)