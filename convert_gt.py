import json
import os
import argparse

def run(annot_path, out_dir):
    # Load annotations json
    with open(annot_path) as f:
        gt_data = f.read()
    annots = json.loads(gt_data)['annotations']
    
    # rope_annots storage
    rope_annots = {}
    # reformat
    for annot in annots:
        # Check if img_id key exists in preds
        img_id = annot['image_id']
        if img_id in rope_annots:
            # append new person data
            rope_annots[img_id].append({
            "person_id": annot['person_id'],
            "category_id": annot['category_id'],
            "bbox": annot['bbox'], 
            "bbox_only": annot['only_bbox'],
            "keypoints": annot['keypoints']
        })
        else:
            rope_annots[img_id] = [{
            "person_id": annot['person_id'],
            "category_id": annot['category_id'],
            "bbox": annot['bbox'], 
            "bbox_only": annot['only_bbox'],
            "keypoints": annot['keypoints']
        }]

    out_path = os.path.join(out_dir, 'rope_gt.json')
    print(f"Saving ROPE MVOR annotations to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(rope_annots, f)
    
    return


def main():
    """
    parse the command line arguments run python convert_gt.py -h to see command line options
    :return:
    """
    parser = argparse.ArgumentParser(description='Visualize the Movenet predictions vs MVOR annotations')
    parser.add_argument(
        '--annot_path',
        type=str,
        default='',
        help='path to original MVOR annotation json'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='.',
        help='path to output directory - defaults to cwd'
    )

    args = parser.parse_args()
    run(args.annot_path, args.out_dir)

if __name__ == '__main__':
  main()

