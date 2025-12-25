import glob
import argparse
import numpy as np
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
# parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

file_filter = args.input

# file_hamer = r"D:\AI\hamer\Testes\SignLanguage\hammer\result_000_*.pkl"
# file_hamer = r"D:\AI\hamer\fight_video\hamer\result_000_*.pkl"

list_hamer_pkl = glob.glob(file_filter)

# checando maximo de pessoas a considerar
# list_people =[]
max_people = 0
num_people = 0
max_lh = 0
max_rh = 0
for i, file in enumerate(list_hamer_pkl): # primeiro loop para poder encontrar os tamanhos de arrays a criar
    with(open(file, "rb")) as f:
        pkl_hamer = pickle.load(f)
    frame = int(os.path.splitext((os.path.basename(file)))[0].split('_')[-1])
    # print('frames: ',frame, ' - personid: ',pkl_hamer["personid"],' - right: ',pkl_hamer["right"], 'poses: ',pkl_hamer["hand_pose"].shape)
    # list_people.append([i,pkl_hamer["personid"],pkl_hamer["right"]])
    num_people = len(pkl_hamer["personid"])
    if num_people > max_people:
        max_people = num_people
    
    num_lh = np.count_nonzero(pkl_hamer["right"] == 0)
    if num_lh > max_lh:
        max_lh = num_lh
    num_rh = np.count_nonzero(pkl_hamer["right"] == 1)
    if num_rh > max_rh:
        max_rh = num_rh


print('frames: ',frame,'max_people: ',max_people, 'max_lh: ',max_lh, 'max_rh: ',max_rh)

# Criando pose numpy array com tamanho ja definido
lh_pose = np.zeros((frame+1,max_lh,15,3,3))
rh_pose = np.zeros((frame+1,max_rh,15,3,3))
lh_global_orient = np.zeros((frame+1,max_lh,1 ,3,3))
rh_global_orient = np.zeros((frame+1,max_rh,1 ,3,3))
hand_export={}
for i, file in enumerate(list_hamer_pkl): # segundo loop para poder encontrar os tamanhos de arrays a criar
    with(open(file, "rb")) as f:
        pkl_hamer = pickle.load(f)
    fr = int(os.path.splitext((os.path.basename(file)))[0].split('_')[-1])
    in_lf = 0 #contar quantidade de vezes que entrou no lf
    in_rh = 0
    for go in range(pkl_hamer["right"].shape[0]):
        if pkl_hamer["right"][go] == 0:
            #trabalhando no left_hand
            in_lf += 1
            # print('go: ',go,'in_lf: ',in_lf,'in_rh: ',in_rh,'frame: ',fr)
            # print('go-in_rh: ',go-in_rh)
            lh_pose[fr][go-in_rh] = pkl_hamer["hand_pose"][go]
            lh_global_orient[fr][go-in_rh] = pkl_hamer["global_orient"][go]
        else:
            #trabalhando no right_hand
            in_rh += 1
            # print('go: ',go,'in_lf: ',in_lf,'in_rh: ',in_rh,'frame: ',fr)
            # print('go-in_lf: ',go-in_lf)
            rh_pose[fr][go-in_lf] = pkl_hamer["hand_pose"][go]
            rh_global_orient[fr][go-in_lf] = pkl_hamer["global_orient"][go]


hand_export["lh_pose"] = lh_pose
hand_export["rh_pose"] = rh_pose
hand_export["lh_global_orient"] = lh_global_orient
hand_export["rh_global_orient"] = rh_global_orient

pkl_out = os.path.join(os.path.dirname(file_filter),"_Hammer_Final_to_convert.pkl")
with(open(pkl_out, "wb")) as f:
    pickle.dump(hand_export, f)