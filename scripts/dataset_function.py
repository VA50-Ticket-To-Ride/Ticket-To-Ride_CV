from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
import os
import cv2 as cv
import numpy as np

def custom_dataset(dataset_path_rel):
    classes_color = {"city": 0, "blu": 1, "whi": 2, "yel": 3, "bla": 4, "ora": 5, "gre": 6, "red": 7, "pur": 8, "gra": 9}
    classes = {"rect": 0, "city": 1}

    imgs = []
    dataset_path = os.path.join(os.getcwd(), dataset_path_rel)
    images_path = os.path.join(dataset_path, "images")
    for image_name in os.listdir(images_path):
        image_path = os.path.join(images_path, image_name)
        img_data = {"file_name": image_path}
        
        img_name_noext = os.path.splitext(image_name)[0]
        img_data["image_id"] = img_name_noext
        
        img_data["width"] = 1576
        img_data["height"] = 1048

        # Read annotations
        annotations = []
        label_name = img_name_noext + ".txt"
        label_path = os.path.join(dataset_path, "labelTxt", label_name)
        with open(label_path, 'r') as label_file:
            lines = [line.rstrip() for line in label_file]
        for line in lines:
            instance = {}
            line_split = line.split(", ")

            if len(line_split) <= 10:
                assert line_split[8][:3] in classes_color
                class_name = line_split[8][:3]
                # class_name = "rect"

                nb_edges = 4
            else:
                class_name = "city"
                nb_edges = 8

            rotated_poly = []
            rotated_poly_np = []
            for _ in range(nb_edges):
                x = float(line_split.pop(0))
                y = float(line_split.pop(0))
                rotated_poly.append(x)
                rotated_poly.append(y)
                rotated_poly_np.append((x, y))
            #class_id = classes[class_name]
            class_id = classes_color[class_name]

            instance["category_id"] = class_id
            
            rotated_poly_np = np.array(rotated_poly_np, dtype=np.float32)
            x, y, w, h = cv.boundingRect(rotated_poly_np)
            bbox = (x, y, w, h)
            instance["bbox"] = bbox
            instance["bbox_mode"] = BoxMode.XYWH_ABS

            instance["segmentation"] = [rotated_poly]

            annotations.append(instance)

        img_data["annotations"] = annotations
        imgs.append(img_data)

    return imgs

def my_dataset():
    return custom_dataset("datasets/dataset_boards")

def inference_dataset():
    return custom_dataset("datasets/dataset_inference")

DatasetCatalog.register("my_dataset", my_dataset)
DatasetCatalog.register("inference_dataset", inference_dataset)

datasets_str = ["my_dataset", "inference_dataset"]
for dataset_str in datasets_str:
    MetadataCatalog.get(dataset_str).evaluator_type = "coco"
    MetadataCatalog.get(dataset_str).thing_classes = ["city", "blu", "whi", "yel", "bla", "ora", "gre", "red", "pur", "gra"]
    MetadataCatalog.get(dataset_str).thing_colors = [(255, 255, 127), (0, 0, 255), (255, 255, 255), (0, 255, 255), (0, 0, 0), (255, 127, 127), (0, 255, 0), (255, 0, 0), (127, 0, 255), (127, 127, 127)]
    MetadataCatalog.get(dataset_str).thing_dataset_id_to_contiguous_id = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
    # MetadataCatalog.get(dataset_str).thing_classes = ["rect", "city"]
    # MetadataCatalog.get(dataset_str).thing_colors = [(0, 0, 255), (255, 255, 127)]
    # MetadataCatalog.get(dataset_str).thing_dataset_id_to_contiguous_id = {0: 0, 1: 1}

# later, to access the data:
# data = DatasetCatalog.get("my_dataset")

# print(data[0]["annotations"][0])
