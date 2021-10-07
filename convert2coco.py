import csv
import json
import os
import pandas as pd
import cv2

item_id = {"trafficcone":0, "can":1, "box":2}

files = ["sub-train-annotations-bbox", "sub-validation-annotations-bbox", "sub-test-annotations-bbox"]
dir = os.path.join(".", "dataset_information")

json_dir = os.path.join(".", "annotation")

for file in files:
    file_path = os.path.join(dir, file+"csv")
    images = []
    annotations = []
    categories = []

    df = pd.read_csv(file_path)
    image_paths = df["ImageID"].unique()
    for id, image_path in enumerate(image_paths):
        #images
        #TODO pathを相対で求めて一撃で終える
        image = cv2.imread(image_path)
        name = os.path.basename(image)
        if "train" in file:
            cv2.imwrite(os.path.join(dir, "train", name))
        elif "validation" in file:
            cv2.imwrite(os.path.join(dir, "validation", name))
        else:
            cv2.imwrite(os.path.join(dir, "test", name))
        h, w = image.shape[:2]
        images.append({
            "file_name" : name,
            "height" : h,
            "width" : w,
            "id" : id,
        })

        #annotations
        image_info = df[df["ImageID"] == image_path]
        for index, row in image_info.iterrows():
            XMin = row["XMin"]
            XMax = row["XMax"]
            YMin = row["YMin"]
            YMax = row["YMax"]
            ClassName = row["ClassName"]
            annotations.append({
                "id" : "10000" + str(id),
                "image_id" : id,
                "iscrowd" : 0,
                "bbox" : [XMin * w, YMin * h, (XMax - XMin)*w, (YMax - YMin)*h],
                "category_id" : item_id[ClassName],
                "area" : (XMax - XMin)*w * (YMax - YMin)*h
            })

    #categories
    for key, value in item_id.items():
        categories.append({
            "supercategory" : key,
            "id" : value,
            "name" : key
        })
    part = ""
    if "train" in file:
        part = "train"
    elif "validation" in file:
        part = "validation"
    else:
        part = "test"
    json_path = os.path.join(json_dir, "instances_" + part + ".json")
    df.to_json("json_path")