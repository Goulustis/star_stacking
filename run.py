from star_align.stack import process_and_stack


if __name__ == "__main__":
    img_dir = "raw_imgs/Stacking 10pm-1"
    work_dir = "pre_proc"
    process_and_stack(img_dir, work_dir)