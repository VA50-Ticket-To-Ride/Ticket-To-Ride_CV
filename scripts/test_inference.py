import torch
import cv2 as cv
import numpy as np
import os
import pickle 
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.transforms import RandomApply, RandomRotation
from detectron2.utils.visualizer import Visualizer
from torchvision import transforms
from PIL import Image

data_transform = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Resize([800, 1203], antialias=None, interpolation=transforms.functional.InterpolationMode.BICUBIC),
    ])

# TODO: Use PIL
# test_img_path = "datasets/dataset_boards/images/b8.png"
test_img_path = "board_4.png"
test_img_noext = os.path.splitext(os.path.split(test_img_path)[1])[0]
test_img = cv.imread(test_img_path, 1)
print(type(test_img))
print(test_img.shape)
# We can probably replace this PIL business with a simple opencv resize, even though the behavior is slightly different
resize_factor = 800 / test_img.shape[0]
new_height = round(test_img.shape[1]*resize_factor)
interpolation = cv.INTER_AREA if resize_factor < 1 else cv.INTER_CUBIC
test_img_resized = cv.resize(test_img, (new_height, 800), interpolation=interpolation)
# pil_image = Image.fromarray(test_img)
# pil_image_resized = pil_image.resize((new_height, 800), Image.BILINEAR)
# test_img_resized = np.asarray(pil_image_resized)
test_tensor = torch.from_numpy(np.copy(test_img_resized.transpose(2, 0, 1)))
test_preprocessed = test_tensor
#test_preprocessed = data_transform(test_tensor)
#test_img = test_img_resized

"""
# Create a config similar to the one used during training
cfg = get_cfg()
cfg.merge_from_file("output/config.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a threshold for this model

# Build the model architecture
model = build_model(cfg)

# Load the weights from your trained model file
model_file = "rect_detector_v2.pth"
checkpointer = DetectionCheckpointer(model)
checkpointer.load(model_file)

# Set the model to evaluation mode
model.eval()
"""
model = torch.jit.load("rect_detector_v2_export/model.ts")

input_data = tuple([{
    "image": test_preprocessed,
    # "file_name": test_img_path,
    # "image_id": test_img_noext,
    # "width": test_img.shape[1],
    # "height": test_img.shape[0]
}])

"""
mapper = DatasetMapper(cfg, True)
mapper.augmentations.augs.append(RandomApply(RandomRotation(angle=(0, 360), expand=False, center=((0.4, 0.4), (0.6, 0.6)))))
print(mapper.augmentations)

import sys
sys.path.append("/mnt/Hdd_ntfs/Documents/GitHub/Ticket-To-Ride_CV/scripts")
from dataset_function import custom_dataset
data_loader = build_detection_test_loader(cfg, "my_dataset", mapper=mapper)
for inputs in data_loader:
    break
"""

"""

# Shape should be [3, 800, 1203]

# Run inference
print(input_data[0]["image"].shape)
#     print(inputs[0]["image"].shape)

inputs[0]["width"] = 1203
inputs[0]["height"] = 800
"""

# outputs = model(input_data)  # image is your input data
# with open('saved_dictionary.pkl', 'wb') as f:
#     pickle.dump(outputs[0], f)
        
with open('saved_dictionary.pkl', 'rb') as f:
    outputs = [pickle.load(f)]

# print(outputs)
#outputs = model(inputs)
# outputs=[{"instances": []}]
#print("Number of detections:" + str(len(outputs[0]["instances"])))

# Convert tensor types to numpy arrays in the right shape
conf_tresh = 0.5
mask_tresh = 0.5

scores = outputs[0]["scores"].detach().numpy()
pred_classes = [x for score, x in zip(scores, outputs[0]["pred_classes"].detach().numpy()) if score >= conf_tresh]
nb_obj = len(pred_classes)

pred_boxes = [x.detach().numpy().reshape(2, 2) for score, x in zip(scores, outputs[0]["pred_boxes"]) if score >= conf_tresh]
assert all(len(x) == 1 for x in outputs[0]["pred_masks"]) # Only one binary mask to describe each object
#pred_polys = [np.asarray(x[0].detach()).reshape(-1, 7, 2) for score, x in zip(scores, outputs[0]["pred_masks"]) if score >= conf_tresh]
def binarize_mask(mask, mask_tresh):
    binary_mask = (mask > mask_tresh).float()

    # Convert the tensor to a NumPy array (0 = no pixel, 1 = pixel)
    binary_mask_np = binary_mask.squeeze().numpy().astype(np.uint8)

    return binary_mask_np

pred_masks = [binarize_mask(mask, mask_tresh) for score, mask in zip(scores, outputs[0]["pred_masks"]) if score >= conf_tresh]

print("Number of detections (all):" + str(len(scores)))
print("Number of detections (>" + str(round(conf_tresh*100)) + "%) :" + str(nb_obj))

# Resize boxes to match original image size
resize_factor = np.divide(test_img.shape, test_img_resized.shape)
for i in range(nb_obj):
    for j in range(2):
        pred_boxes[i][j, 0] *= resize_factor[1]
        pred_boxes[i][j, 1] *= resize_factor[0]

# Visualize the predictions (optional)
#v = Visualizer(test_img)
#v = Visualizer(inputs[0]["image"].numpy().transpose((1, 2, 0)))
#v = v.draw_instance_predictions(outputs[0]["instances"])
#v = v.draw_instance_predictions(outputs[0])

def rect_rel_to_abs(box, rect_rel):
    box_width = box[1][0] - box[0][0]
    box_height = box[1][1] - box[0][1]
    
    (x, y), (width, heigth), angle = rect_rel
    x_abs = box[0][0] + box_width*x
    y_abs = box[0][1] + box_height*y
    width_abs = box_width*width
    height_abs = box_height*heigth
    
    rect_abs = (x_abs, y_abs), (height_abs, width_abs), angle

    print(box)
    print(rect_rel)
    print(rect_abs)

    return rect_abs

rects = [None]*nb_obj
for i in range(nb_obj):
    box_int = np.intp(np.round(pred_boxes[i]))
    box_width = box_int[1][0] - box_int[0][0]
    box_height = box_int[1][1] - box_int[0][1]
    mask_x = box_int[0][0]
    mask_y = box_int[0][1]

    # Resize the mask
    resized_mask = cv.resize(pred_masks[i], (box_width, box_height))

    # Erode mask
    morph_kernel = np.ones((3, 3), np.uint8)
    resized_mask = cv.erode(resized_mask, morph_kernel, iterations = 1)

    # Find contours
    contours, _ = cv.findContours(resized_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contour = max(contours, key=cv.contourArea)

    (center_x, center_y), (width, height), angle = cv.minAreaRect(contour)
    rect = ((center_x+mask_x, center_y+mask_y), (width, height), angle)
    rbox = cv.boxPoints(rect)
    rbox = np.intp(rbox)
    
    # Draw mask on image
    # img_slice = test_img[mask_y:mask_y + box_height, mask_x:mask_x + box_width]
    # img_slice[np.where(resized_mask)] = [127, 255, 127]  # Set white color (255, 255, 255) for the masked pixels

    # rect_rel = cv.minAreaRect(pred_polys[i][0])

    # rects[i] = rect_rel_to_abs(pred_boxes[i], rect_rel)
    # print(poly)

    #print(box_int)
    #cv.rectangle(test_img, pt1=box_int[0], pt2=box_int[1], color=(255, 255, 127), thickness=2)
    # rbox = np.intp(cv.boxPoints(rects[i]))
    # print(rbox)
    cv.drawContours(test_img, [rbox], 0, color=(255, 255, 127), thickness=2)

cv.namedWindow('display', cv.WINDOW_NORMAL)
# cv.imshow("display", v.get_image())
cv.imshow("display", test_img)
cv.waitKey(0)
#"""

# img_machine = inputs[0]["image"].numpy().transpose((1, 2, 0))
# img_hand = input_data[0]["image"].numpy().transpose((1, 2, 0))
# concat = np.concatenate((img_machine, img_hand), axis=1)
# print(test_img[0, 337])
# print(img_machine[0, 337])
# print(img_hand[0, 337])
# cv.imshow('display', img_machine)
# #cv.imshow('display', img_hand)
# cv.waitKey(0)
