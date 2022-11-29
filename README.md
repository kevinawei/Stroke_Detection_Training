# Stroke_Detection_Training



## Data
[Link to google drive with training images and dataset instructions](https://drive.google.com/drive/folders/1Pdd8phPOgasBIXaX9CAk65HhruBgEjii?usp=sharing)

![Example of training data](/screenshots/data1.jpg)

![](/screenshots/data2.jpg)
### Script to reformat XML files to be processed with labels

"""
This script reformats XML files labelled with CVAT to
contain the attribute pose = 'unspecified'.
It also changes bounding box float values to integers.
"""

import xmltodict
import pprint
import os

pp = pprint.PrettyPrinter(indent=2)

SOURCE_PATH = "./annotations-train-raw"
DEST_PATH = "./annotations-train-clean"

annotations = os.listdir({SOURCE_PATH})
valid_data_count = 0

for annotation in annotations:
    xml = open(f"{SOURCE_PATH}/{annotation}", "r")
    org_xml = xml.read()
    dict_xml = xmltodict.parse(org_xml, process_namespaces=True)
    # Not a jpg file
    if not dict_xml['annotation']['filename'].endswith('.jpg'):
        continue
    # If Annotation has bounding box
    try:
        for obj in dict_xml['annotation']['object']:
            obj['pose'] = "Unspecified"
            for key in obj['bndbox']:
                obj['bndbox'][key] = int(float(obj['bndbox'][key]))

        out = xmltodict.unparse(dict_xml, pretty=True)
        out = out.split("\n", 1)[1]

        valid_data_count += 1
        with open(f"{DEST_PATH}/{annotation}", 'wb') as file:
            file.write(out.encode('utf-8'))
    except:
        print(f"{annotation} was not labelled")

# Train - 419/603 (Ferdz & WenJin)
# Validation - 220/313 (Kevin)
# Total Data: 639
print(f"Total Cleaned: {valid_data_count}/{len(annotations)}")



## Results
![Tensorboard Graphs](/screenshots/learning_rate.png)
![](/screenshots/epoch_cls_loss.png)
![](/screenshots/epoch_det_loss.png)
![](/screenshots/epoch_loss.png)
![](/screenshots/time_series_loss.png)



## Testing