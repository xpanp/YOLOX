import json

with open("train_annos_1-60.json", "r", encoding='utf-8') as f:
    datas = json.load(f)

licenses = [
    {
        "id": 1,
        "name": "GNU General Public License v3.0",
        "url": "https://github.com/zhiqwang/yolov5-rt-stack/blob/master/LICENSE"
    }
]

info = {
    "year": 2021,
    "version": "1.0",
    "description": "For food object detection",
    "date_created": "2021"
}

cocoTrain = {
    "info" : info
}
cocoTrain["licenses"] = licenses
cocoTrain["type"] = "instances"
cocoTrain["images"] = []
cocoTrain["annotations"] = []
cocoTrain["categories"] = []

cocoTest = {
    "info" : info
}
cocoTest["licenses"] = licenses
cocoTest["type"] = "instances"
cocoTest["images"] = []
cocoTest["annotations"] = []
cocoTest["categories"] = []

def annFill(data, ann_id, image_id):
    ann = {"id": ann_id, "image_id": image_id}
    bbox = data["bbox"]
    cocoBbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
    ann["bbox"] = cocoBbox
    ann["category_id"] = data["category"]
    ann["iscrowd"] = 0
    ann["area"] = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
    return ann

def imgFill(data, id, file_name):
    image = {"date_captured": "2021"}
    image["file_name"] = file_name
    image["id"] = id
    image["height"] = data["image_height"]
    image["width"] = data["image_width"]
    return image

pos = 1
pre_name = ''
file_name = ''
train_img_id = 1
train_ann_id = 1
test_img_id = 1
test_ann_id = 1
flag = True
pre_flag = True
for data in datas:
    file_name = data['name']
    if pre_name == file_name:
        flag = pre_flag
    else:
        pre_name = file_name
        if pos % 10 == 0:
            flag = False
            pre_flag = flag
        else:
            flag = True
            pre_flag = flag
        pos += 1
        if flag == True:
            image = imgFill(data, train_img_id, file_name)
            train_img_id += 1
            cocoTrain["images"].append(image)
        else:
            image = imgFill(data, test_img_id, file_name)
            test_img_id += 1
            cocoTest["images"].append(image)
    if flag == True:
        ann = annFill(data, train_ann_id, train_img_id-1)
        train_ann_id += 1
        cocoTrain["annotations"].append(ann)
    else:
        ann = annFill(data, test_ann_id, test_img_id-1)
        test_ann_id += 1
        cocoTest["annotations"].append(ann)

for i in range(60):
    categorie = {"id": i+1, "name": str(i), "supercategory":str(i)}
    cocoTrain["categories"].append(categorie)
    cocoTest["categories"].append(categorie)

with open("train_food.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(cocoTrain, ensure_ascii=False))

with open("test_food.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(cocoTest, ensure_ascii=False))