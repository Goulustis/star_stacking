import cv2
import numpy as np
import os.path as osp
import glob
from tqdm import tqdm
from loguru import logger

from star_align.utils import parallel_map
from star_align.light import remove_light_pollution
from star_align.matching.match_track import build_track

def warp_img(img:np.ndarray,
             src_pts:np.ndarray,
             dst_pts:np.ndarray):
    h, w = img.shape[:2]

    hom, inlier_msk = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return cv2.warpPerspective(img, hom, (w, h)).astype(img.dtype)

def avg_imgs(imgs):
    st_img = imgs[0].astype(np.float32)

    for img in tqdm(imgs[1:], desc="averaging imgs"):
        st_img = st_img + img.astype(np.float32)
        
    return st_img / len(imgs)

def warp_stack(img_dir:str, track:np.ndarray, out_f:str):
    img_fs = sorted(glob.glob(osp.join(img_dir, "*image*.png")))

    ref_img = cv2.imread(img_fs[0], cv2.IMREAD_UNCHANGED)
    ref_pts = track[0]
    warpped_imgs = []
    for i in tqdm(range(1, len(img_fs)), desc="warpping imgs"):
        pts = track[i]
        val_cond = ~np.isnan(pts[:, 0])
        w_img = warp_img(cv2.imread(img_fs[i], cv2.IMREAD_UNCHANGED), pts[val_cond], ref_pts[val_cond])
        warpped_imgs.append(w_img)
    
    stacked_img = avg_imgs([ref_img] + warpped_imgs)
    stacked_img = stacked_img
    stacked_img = stacked_img.astype(np.uint16)
    cv2.imwrite(out_f, stacked_img)
        

def naive_stack(img_dir:str, out_f:str):
    img_fs = sorted(glob.glob(osp.join(img_dir, "*.png")))
    imgs = parallel_map(cv2.imread, img_fs, show_pbar=True, desc="loading images")

    stacked_img = avg_imgs(imgs)
    stacked_img = stacked_img/255 * (2**16 - 1)
    stacked_img = stacked_img.astype(np.uint16)
    cv2.imwrite(out_f, stacked_img)


def process_and_stack(img_dir:str, work_dir:str):
    
    logger.info('removing light pollution')
    corr_img_dir = remove_light_pollution(img_dir, work_dir)

    logger.info("finding correspondences via tracking")
    track_f = osp.join(corr_img_dir, "track.npy")
    track = build_track(corr_img_dir, track_f)
    
    out_f = osp.join(work_dir, f"stack_{osp.basename(img_dir)}")
    warp_stack(corr_img_dir, track, out_f)
