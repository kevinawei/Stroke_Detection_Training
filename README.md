# Stroke_Detection_Training



## Data
[Link to google drive with training images and dataset instructions](https://drive.google.com/drive/folders/1Pdd8phPOgasBIXaX9CAk65HhruBgEjii?usp=sharing)

### Examples of training/validation data before annotation (see more details on requirements and the annotation process in the instruction document located at the above link)
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

### Train - 419/603 (Ferdz & WenJin)
### Validation - 220/313 (Kevin)
### Total Data: 639
print(f"Total Cleaned: {valid_data_count}/{len(annotations)}")



## Results
[Link to google colab](https://colab.research.google.com/drive/19RhGfiKIkM0KLWVSCt6hnT3fcXBlTCMz?authuser=3)


We trained our EfficientDet-Lite3 Model with a batch size of 4 over 100 epochs

EfficientDet-Lite3 was selected over other more lightweight alternatives because the slightly increased latency wouldn't seem to effect the usage of an app of this nature (stroke detection symptoms would last a long time and the latency is low enough that it would be able to detect changes within a few seconds at worst)

![Tensorboard Graphs](/screenshots/learning_rate.png)
![](/screenshots/epoch_cls_loss.png)
![](/screenshots/epoch_det_loss.png)
![](/screenshots/epoch_loss.png)
![](/screenshots/time_series_loss.png)



## Testing

### Testing result screenshots
![](/screenshots/1.png)
![](/screenshots/7.png)
![](/screenshots/2.png)
![](/screenshots/3.png)
![](/screenshots/4.png)
![](/screenshots/5.png)
![](/screenshots/6.png)
![](/screenshots/screenshot1.png)
![](/screenshots/screenshot2.png)
![](/screenshots/screenshot3.png)