import os


def addSuperCategoryandFixCategoryIDs(json_path_In, json_path_Out):

    import json

    for filename in os.listdir(json_path_In):
        if filename.endswith(".json"):
            file_path = os.path.join(json_path_In, filename)

    #open any.json file in folder
    with open(file_path, "r") as f:
        data = json.load(f)

    # Add 'supercategory' if missing
    for cat in data.get("categories", []):
        cat["supercategory"] = "cell"

    # Fix category IDs
    for cat in data["categories"]:
        cat["id"] -= 1

    for ann in data["annotations"]:
        ann["category_id"] -= 1

    with open(json_path_Out, "w") as f:
        json.dump(data, f)

