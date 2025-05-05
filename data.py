import json
import cv2
import numpy as np

def load_data(datadir, split="train"):
    with open(f"{datadir}/transforms_{split}.json", "r") as file:
        meta = json.load(file)

    images = []
    poses = []
    for frame in meta['frames']:
        image_path = f"{datadir}/{frame['file_path'][2:]}.png"
        print(image_path)
        img = cv2.imread(image_path)
        img = img[..., ::-1]
        images.append(img/255.0)
        poses.append(np.array(frame['transform_matrix']))
        #print("frame\n", frame['transform_matrix'])
        #print("array\n", poses[0])
        #cv2.imshow("img", images[0])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #break
    H,W = images[0].shape[:2]
    focal = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])
    return np.array(images), np.array(poses), H, W, focal

def load_eval_data(datadir, index, split="train"):
    with open(f"{datadir}/transforms_{split}.json", "r") as file:
        meta = json.load(file)

    frame = meta['frames'][index]
    image_path = f"{datadir}/{frame['file_path'][2:]}.png"
    print(image_path)

    image = cv2.imread(image_path)
    pose =np.array(frame['transform_matrix'])

    H,W = image.shape[:2]
    focal = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])
    return image, pose, H, W, focal

if __name__=="__main__":
    datadir = "nerf_recon_dataset/nerf_synthetic/hotdog"
    load_data(datadir)