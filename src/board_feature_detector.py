from cell import *
from node import *

import os
import math
import torch
import torchvision # Import needed to load the scripted model
from scipy.optimize import linear_sum_assignment

class BoardFeatureDetector:
    """
    Docs dump:
    Contours detection
    https://docs.opencv.org/4.8.0/d4/d73/tutorial_py_contours_begin.html
    https://docs.opencv.org/4.8.0/dd/d49/tutorial_py_contour_features.html
    https://docs.opencv.org/4.8.0/d9/d8b/tutorial_py_contours_hierarchy.html

    Collision
    https://stackoverflow.com/questions/56100547/how-do-i-check-collision-between-a-line-and-a-rect-in-pygame
    """
    def __init__(self, board, debug=False):
        self.debug = debug

        # Use a deep learning model to detect train tracks and cities 
        pred_classes, pred_boxes, pred_masks = self.run_inference(board, box_conf_tresh=0.5, mask_conf_tresh=0.5)

        # Get number of objects
        nb_obj = len(pred_classes)

        if self.debug:
            debug_img_boxes = board.copy()
            debug_img_masks = board.copy()
            debug_img_masks_eroded = board.copy()
            debug_img_rboxes = board.copy()
            debug_img_hitboxes = board.copy()
            debug_img_hitboxes_black = np.zeros_like(board)
            debug_img_indexes = debug_img_hitboxes_black.copy()
        
        # Process inference data to create nodes (cities) and cells (train tracks)
        nodes = []
        cells = []
        for i in range(nb_obj):
            # Cell
            if pred_classes[i] == 0:
                # Get some useful coordinates related to the upright bounding box
                box_int = np.intp(np.round(pred_boxes[i]))
                box_width = box_int[1][0] - box_int[0][0]
                box_height = box_int[1][1] - box_int[0][1]
                mask_x = box_int[0][0]
                mask_y = box_int[0][1]

                # Resize the square mask to the bounding box size
                resized_mask = cv.resize(pred_masks[i], (box_width, box_height))

                # Erode mask
                morph_kernel = np.ones((3, 3), np.uint8)
                eroded_mask = cv.erode(resized_mask, morph_kernel, iterations=1)

                # Find contours
                contours, _ = cv.findContours(eroded_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                # Keep contour with the biggest area
                contour = max(contours, key=cv.contourArea)

                # Find smallest fitting rotated rectangle and its corresponding rotated bounding box
                (center_x, center_y), (width, height), angle = cv.minAreaRect(contour)
                rect = ((center_x+mask_x, center_y+mask_y), (width, height), angle)
                rbox = np.intp(cv.boxPoints(rect))

                # Create cell object and add it to the list
                cell = Cell(len(cells), rect, rbox)
                cells.append(cell)

                if debug:
                    # debug_img_boxes
                    cv.rectangle(debug_img_boxes, pt1=box_int[0], pt2=box_int[1], color=(255, 255, 127), thickness=2)
                    # debug_img_masks
                    img_slice = debug_img_masks[mask_y:mask_y + box_height, mask_x:mask_x + box_width]
                    img_slice[np.where(resized_mask)] = [127, 127, 255]
                    # debug_img_masks_eroded
                    img_slice = debug_img_masks_eroded[mask_y:mask_y + box_height, mask_x:mask_x + box_width]
                    img_slice[np.where(eroded_mask)] = [127, 127, 255]
                    # debug_img_rboxes
                    cv.drawContours(debug_img_rboxes, [rbox], 0, color=(255, 255, 127), thickness=2)
                    # debug_img_hitboxes
                    cell.draw_hitbox(debug_img_hitboxes)
                    # debug_img_hitboxes_black
                    cell.draw_hitbox(debug_img_hitboxes_black)
                    # debug_img_indexes
                    cell.draw(debug_img_indexes)

            # Node
            elif pred_classes[i] == 1:
                # Find center of bounding box
                box = pred_boxes[i]
                box_center_x = round((box[0][0] + box[1][0]) / 2)
                box_center_y = round((box[0][1] + box[1][1]) / 2)

                # Create node object and add it to the list
                node = Node((box_center_x, box_center_y))
                nodes.append(node)

                if debug:
                    box_int = np.intp(np.round(box))
                    # debug_img_boxes
                    cv.rectangle(debug_img_boxes, pt1=box_int[0], pt2=box_int[1], color=(127, 255, 127), thickness=2)
                    # debug_img_rboxes
                    node.draw_hitbox(debug_img_rboxes, radius=12)
                    # debug_img_hitboxes
                    node.draw_hitbox(debug_img_hitboxes, color=(255, 255, 127))
                    # debug_img_hitboxes_black
                    node.draw_hitbox(debug_img_hitboxes_black, color=(255, 255, 127))

            # Invalid class ID
            else:
                raise Exception("Invalid inference class ID: " + str(pred_classes[i]))
            
        if self.debug:
            cv.imwrite('debug/2_inference_bounding_boxes.png', debug_img_boxes)
            cv.imwrite('debug/3_inference_masks.png', debug_img_masks)
            cv.imwrite('debug/4_inference_masks_eroded.png', debug_img_masks_eroded)
            cv.imwrite('debug/5_inference_rotated_bboxes.png', debug_img_rboxes)
            cv.imwrite('debug/6_objects_hitboxes.png', debug_img_hitboxes)
            cv.imwrite('debug/7_objects_hitboxes_black.png', debug_img_hitboxes_black)
            cv.imwrite('debug/8_objects_indexes.png', debug_img_indexes)

        # Build collision map
        # node-cell
        for node in nodes:
            node.search_collisions(cells)

        # cell-cell
        for i, cell in enumerate(cells):
            cell.search_collisions(cells[i+1:])

        if self.debug:
            debug_img_links = np.zeros_like(board)
            for cell in cells:
                cell.draw_links(debug_img_links)
            cv.imwrite('debug/9_objects_links.png', debug_img_links)
        
        # Solve multiple connectivity issues
        # Create cost array
        costs = np.full((2*len(cells), 2*len(cells)), fill_value=1e10, dtype=np.float32)
        problematic_indexes = []
        for i, cell in enumerate(cells):
            links_dicts = cell.get_links_dicts()
            for side, links_dict in enumerate(links_dicts):
                row_index = 2*i + side
                if len(links_dict) > 1:
                    problematic_indexes.append(row_index)

                for col_index, dist in links_dict.items():
                    costs[row_index, col_index] = dist

        # Apply the Hungarian algorithm to minimize the total distance
        row_ind, col_ind = linear_sum_assignment(costs)

        # Remove links from the cell sides with several links using the above
        for problematic_index in problematic_indexes:
            cell_index = problematic_index // 2
            cell_side = problematic_index % 2

            best_link_index = col_ind[np.where(row_ind==problematic_index)[0][0]] // 2
            cells[cell_index].keep_best_link(cell_side, best_link_index)

        if self.debug:
            debug_img_links_unique = np.zeros_like(board)
            for cell in cells:
                cell.draw_links(debug_img_links_unique)
            cv.imwrite('debug/10_objects_links_unique.png', debug_img_links_unique)
    
    def run_inference(self, img, box_conf_tresh, mask_conf_tresh):
        # Small utility function to convert a confidence tensor mask to a binary numpy one
        def binarize_mask(mask, mask_conf_tresh):
            # Binarize tensor according to treshold
            binary_mask = (mask > mask_conf_tresh).float()

            # Convert the tensor to a NumPy array (0 = no pixel, 1 = pixel)
            binary_mask_np = binary_mask.squeeze().numpy().astype(np.uint8)

            return binary_mask_np
        
        # Resize input image. Note that the model was trained using PIL's resize (bilinear) which is slightly different
        resize_factor = 800 / img.shape[0]
        new_height = round(img.shape[1]*resize_factor)
        interpolation = cv.INTER_AREA if resize_factor < 1 else cv.INTER_CUBIC
        img_resized = cv.resize(img, (new_height, 800), interpolation=interpolation)

        # Convert it to tensor
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1))

        # Convert input to right format
        inputs = tuple([{
            "image": img_tensor,
        }])
        
        # Load model
        model = torch.jit.load("resources/rect_detector_v2.ts")
        
        # Run inference
        outputs = model(inputs)
        
        # import pickle
        # with open('ignore/saved_dictionary.pkl', 'wb') as f:
        #     pickle.dump(outputs[0], f)

        # import pickle
        # with open('ignore/saved_dictionary.pkl', 'rb') as f:
        #     outputs = [pickle.load(f)]

        # Extract scores to treshold other data
        scores = outputs[0]["scores"].detach().numpy()

        # Extract predicted classes, boxes and masks
        assert all(len(x) == 1 for x in outputs[0]["pred_masks"]) # Only one binary mask to describe each object
        pred_classes = [x for score, x in zip(scores, outputs[0]["pred_classes"].detach().numpy()) if score >= box_conf_tresh]
        pred_boxes = [x.detach().numpy().reshape(2, 2) for score, x in zip(scores, outputs[0]["pred_boxes"]) if score >= box_conf_tresh]
        pred_masks = [binarize_mask(mask, mask_conf_tresh) for score, mask in zip(scores, outputs[0]["pred_masks"]) if score >= box_conf_tresh]
        
        # Get number of objects
        nb_obj = len(pred_classes)

        # Resize boxes to match original image size
        resize_factor = np.divide(img.shape, img_resized.shape)
        for i in range(nb_obj):
            for j in range(2):
                pred_boxes[i][j, 0] *= resize_factor[1]
                pred_boxes[i][j, 1] *= resize_factor[0]

        if self.debug:
            print("Number of detections (all):" + str(len(scores)))
            print("Number of detections (>" + str(round(box_conf_tresh*100)) + "%) :" + str(nb_obj))

            os.makedirs("debug", exist_ok=True)
            cv.imwrite('debug/0_board.png', img)
            cv.imwrite('debug/1_inference_resized.png', img_resized)

        return pred_classes, pred_boxes, pred_masks
