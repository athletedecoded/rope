import argparse
import os
import json

import cv2
import glob

CAMMA_SKELETON = [
  [0, 1], 
  [1, 3], 
  [3, 7], 
  [3, 5], 
  [7, 9], 
  [1, 2],
  [2, 4], 
  [2, 6], 
  [6,8], 
  [4, 5]
]

def draw_person(image, person, cmap, detection_threshold = 0.1):
    """
    image: np image array
    person: rope formatted person instance
    cmap: colour map [keypoints, edges]
    detection_threshold: minimum confidence score to include keypoint
    """
    kypt_color, edge_color = cmap
    # Draw all the landmarks
    if person['bbox_only'] == 0:
        kypts = [person['keypoints'][i:i+3] for i in range(0, len(person['keypoints']), 3)]
        # Draw skeleton if all keypoints detected above threshold
        if min([score for [x,y,score] in kypts]) > detection_threshold:
            for pt in kypts:
                x,y,score = int(pt[0]), int(pt[1]), pt[2]
                if score > detection_threshold:
                    cv2.circle(image, (x,y), 2, kypt_color, 4)
            # Draw the edges
            for edge in CAMMA_SKELETON:
                x1y1 = (int(kypts[edge[0]][0]), int(kypts[edge[0]][1]))
                x2y2 = (int(kypts[edge[1]][0]), int(kypts[edge[1]][1]))
                cv2.line(image, x1y1, x2y2, edge_color, 2)
    # Draw the bbox
    start_point = (int(person['bbox'][0]),int(person['bbox'][1]))
    end_point = (int(person['bbox'][0] + person['bbox'][2]), int(person['bbox'][1] + person['bbox'][3]))
    cv2.rectangle(image, start_point, end_point, edge_color, 2)
    id_text = 'id = ' + str(person['person_id'])
    cv2.putText(image, id_text, start_point, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    return image
    

def visualize_preds(annots_path, preds_path, day, cam):
    """
    annots_path: path to MVOR ground truth annotations json [string]
    preds_path: path to MMP predictions json [string]
    day: day num [int]
    cam: cam num [int]
    """

    # Load annotations json
    with open(annots_path) as f:
        gt_data = f.read()
    gt_annots = json.loads(gt_data)

    # Load predictions json
    with open(preds_path) as f:
        preds_data = f.read()
    preds = json.loads(preds_data)
  
    # Walk images
    dir_path = f'mvor/day{day}/cam{cam}'
    data_path = os.path.join(os.getcwd(), dir_path,'*png') 
    files = glob.glob(data_path)
    print(f'Found {len(files)} images in {dir_path}')

    # Set file display
    cv2.namedWindow("ROPE", cv2.WINDOW_NORMAL)

    for frame in files:
        image = cv2.imread(frame)

        img_name = frame.split('/')[-1]
        img_num, ext = img_name.split('.')
        img_id = f'{day}00{cam}0{img_num}'
        print(f'Image: {img_id}')

        # If prediction exists for image, plot data
        if img_id in preds:
            img_preds = preds[img_id]
            # red keypoints, green edges
            cmap = [(0,0,255),(0,255,0)]
            # For each person
            for person in img_preds:
                image = draw_person(image, person, cmap)
        # If gt_annot exists for the image, plot data
        if img_id in gt_annots:
            img_gt = gt_annots[img_id]
            # white keypoints, yellow edges
            cmap = [(255,255,255),(0,255,255)]
            # For each person
            for person in img_gt:
                image = draw_person(image, person, cmap)
        # Stop the program if the ESC key is pressed else toggle on key
        key = cv2.waitKey(0)
        if key == 27:
            break
        else:
            cv2.imshow("ROPE", image)

    cv2.destroyAllWindows()


def main():
    """
    parse the command line arguments run python viz_preds.py -h to see command line options
    :return:
    """
    parser = argparse.ArgumentParser(description='Visualize the Movenet predictions vs MVOR annotations')
    parser.add_argument(
        '--annots',
        type=str,
        default='',
        help='path to `rope_gt.json`'
    )
    parser.add_argument(
        '--preds',
        type=str,
        default='',
        help='path to the Movenet predictions json'
    )
    parser.add_argument(
        '--day',
        type=str,
        default="",
        help='day number'
    )
    parser.add_argument(
        '--cam',
        type=str,
        default="",
        help='camera number'
    )

    args = parser.parse_args()

    visualize_preds(args.annots, args.preds, args.day, args.cam)

if __name__ == '__main__':
  main()