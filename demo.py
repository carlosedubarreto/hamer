from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import pickle
import subprocess
import shutil

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full


import sys
# 1. Get the current working directory (where the script is being executed from)
script_file_path = os.path.abspath(__file__)
# 2. Extract the directory name from the full path
script_path = os.path.dirname(script_file_path)
# 2. Add the path to sys.path
# sys.path is a list of strings that determines the module search path.
if script_path not in sys.path:
    sys.path.append(script_path)


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

def video_to_frames_ffmpeg(input_video, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Pattern for output filenames (e.g., frame_0001.jpg)
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    
    # Command Breakdown:
    # -i: Input file
    # -vf fps=1: (Optional) Set frame rate. Remove this to extract ALL frames.
    # -q:v 2: Quality (2 is high quality for JPG).
    command = [
        "ffmpeg",
        "-i", input_video,
        "-q:v", "2",  # Best quality for JPG (range 2-31, lower is better)
        output_pattern
    ]
    
    # Run the command efficiently
    subprocess.run(command, check=True)
    print("Extraction complete.")

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--input_video', type=str,  help='Video to process')
    # parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()

    # Download and load checkpoints
    # download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    # os.makedirs(args.out_folder, exist_ok=True)
    if os.path.exists(args.out_folder):
        shutil.rmtree(args.out_folder)
        os.mkdir(args.out_folder)
    else:
        os.mkdir(args.out_folder)

    
    #get the path to this script
    path = os.path.abspath(__file__)
    

    img_seq = os.path.join(os.path.dirname(path),'img_seq')
    if os.path.exists(img_seq):
        shutil.rmtree(img_seq)
        os.mkdir(img_seq)
    else:
        os.mkdir(img_seq)
    video_to_frames_ffmpeg(args.input_video, img_seq)


    # Get all demo images ends with .jpg or .png
    # img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    img_paths = [img for end in args.file_type for img in Path(img_seq).glob(end)]



    # Iterate over all images in folder
    for frame,img_path in enumerate(img_paths):
        batch_pkl = {}
        pkl_out = {}
        print('Frame: ',str(frame).zfill(4))
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        # is_left = [] #WAS  0: left, 1: left, now I've created a setting for each hand, if it exists will be 1 otherwise 0
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                # is_left.append(1)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                # is_left.append(0)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        # left = np.stack(is_left)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for b,batch in enumerate(dataloader):
            
            print('batch: ',str(b).zfill(4))

            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            img_fn, _ = os.path.splitext(os.path.basename(img_path))



            # # batch_pkl['img'] = batch['img'].cpu().numpy()
            # # if f == 0 :
            # batch_pkl['personid'] = batch['personid'].cpu().numpy()
            # batch_pkl['right'] = batch['right'].cpu().numpy()
            # # else:
            # #     batch_pkl['personid'] = np.vstack((batch_pkl['personid'], batch['personid'].cpu().numpy()))
            # #     batch_pkl['right'] = np.vstack((batch_pkl['right'],batch['right'].cpu().numpy()))
            # # # batch_pkl['box_center'] = batch['box_center'].cpu().numpy()
            # # # batch_pkl['box_size'] = batch['box_size'].cpu().numpy()
            # # # batch_pkl['img_size'] = batch['img_size'].cpu().numpy()



            # if f == 0 :
            # pkl_out['is_left'] = is_left
            # pkl_out['is_right'] = is_right
            pkl_out['personid'] = batch['personid'].cpu().numpy()
            pkl_out['right'] = batch['right'].cpu().numpy()
            pkl_out['pred_cam'] = out['pred_cam'].cpu().numpy()
            pkl_out['global_orient'] = out['pred_mano_params']['global_orient'].cpu().numpy()
            pkl_out['hand_pose'] = out['pred_mano_params']['hand_pose'].cpu().numpy()
            pkl_out['betas'] = out['pred_mano_params']['betas'].cpu().numpy()
            pkl_out['pred_cam_t'] = out['pred_cam_t'].cpu().numpy()
            pkl_out['focal_length'] = out['focal_length'].cpu().numpy()
            pkl_out['pred_keypoints_3d'] = out['pred_keypoints_3d'].cpu().numpy()
            pkl_out['pred_vertices'] = out['pred_vertices'].cpu().numpy()
            # else:
            #     pkl_out['pred_cam'] = np.vstack((pkl_out['pred_cam'],out['pred_cam'].cpu().numpy()))
            #     pkl_out['global_orient'] = np.vstack((pkl_out['global_orient'],out['pred_mano_params']['global_orient'].cpu().numpy()))
            #     pkl_out['hand_pose'] = np.vstack((pkl_out['hand_pose'],out['pred_mano_params']['hand_pose'].cpu().numpy()))
            #     pkl_out['betas'] = np.vstack((pkl_out['betas'],out['pred_mano_params']['betas'].cpu().numpy()))
            #     pkl_out['pred_cam_t'] = np.vstack((pkl_out['pred_cam_t'],out['pred_cam_t'].cpu().numpy()))
            #     pkl_out['focal_length'] = np.vstack((pkl_out['focal_length'],out['focal_length'].cpu().numpy()))
            #     pkl_out['pred_keypoints_3d'] = np.vstack((pkl_out['pred_keypoints_3d'],out['pred_keypoints_3d'].cpu().numpy()))
            #     pkl_out['pred_vertices'] = np.vstack((pkl_out['pred_vertices'],out['pred_vertices'].cpu().numpy()))
            # pkl_out['pred_keypoints_2d'] = out['pred_keypoints_2d'].cpu().numpy()
            
            
            # batch_pkl_out_path = os.path.join(args.out_folder, f'batch_{str(b).zfill(3)}_{str(frame).zfill(5)}.pkl')
            # with open(batch_pkl_out_path, 'wb') as f:
            #     pickle.dump(batch_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
            pkl_out_path = os.path.join(args.out_folder, f'result_{str(b).zfill(3)}_{str(frame).zfill(5)}.pkl')
            with open(pkl_out_path, 'wb') as f:
                pickle.dump(pkl_out, f, protocol=pickle.HIGHEST_PROTOCOL)
            


            # # Render the result
            # batch_size = batch['img'].shape[0]
            # for n in range(batch_size):
            #     # # Get filename from path img_path
            #     img_fn, _ = os.path.splitext(os.path.basename(img_path))
            #     person_id = int(batch['personid'][n])
            #     # white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
            #     # input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
            #     # input_patch = input_patch.permute(1,2,0).numpy()

            #     # regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
            #     #                         out['pred_cam_t'][n].detach().cpu().numpy(),
            #     #                         batch['img'][n],
            #     #                         mesh_base_color=LIGHT_BLUE,
            #     #                         scene_bg_color=(1, 1, 1),
            #     #                         )

            #     # if args.side_view:
            #     #     side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
            #     #                             out['pred_cam_t'][n].detach().cpu().numpy(),
            #     #                             white_img,
            #     #                             mesh_base_color=LIGHT_BLUE,
            #     #                             scene_bg_color=(1, 1, 1),
            #     #                             side_view=True)
            #     #     final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
            #     # else:
            #     #     final_img = np.concatenate([input_patch, regression_img], axis=1)

            #     # cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

            #     # Add all verts and cams to list
            #     verts = out['pred_vertices'][n].detach().cpu().numpy()
            #     is_right = batch['right'][n].cpu().numpy()
            #     verts[:,0] = (2*is_right-1)*verts[:,0]
            #     cam_t = pred_cam_t_full[n]
            #     all_verts.append(verts)
            #     all_cam_t.append(cam_t)
            #     all_right.append(is_right)

            #     # Save all meshes to disk
            #     if args.save_mesh:
            #         camera_translation = cam_t.copy()
            #         tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
            #         tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        # if args.full_frame and len(all_verts) > 0:
        #     misc_args = dict(
        #         mesh_base_color=LIGHT_BLUE,
        #         scene_bg_color=(1, 1, 1),
        #         focal_length=scaled_focal_length,
        #     )
        #     cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

        #     # Overlay image
        #     input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
        #     input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
        #     input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

        #     cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

if __name__ == '__main__':
    main()
