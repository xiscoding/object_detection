import supervision as sv
from tqdm import tqdm
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM #segment 
from autodistill_grounding_dino import GroundingDINO #bbox
from autodistill_yolov8 import YOLOv8
from autodistill.detection import CaptionOntology
from autodistill_utils.display_utils import display_autolabeld_dataset, display_img_sample

HOME = "/home/xdoestech/Desktop/object_detection"
IMAGE_DIR_PATH = f"{HOME}/scraped_images/yield_sign_3"
ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/autodistill_all/yield3_dataset/train/labels"
IMAGES_DIRECTORY_PATH = f"{HOME}/autodistill_all/yield3_dataset/train/images"
DATA_YAML_PATH = f"{HOME}/autodistill_all/yield3_dataset/data.yaml"
DATASET_DIR_PATH = f"{HOME}/autodistill_all/yield3_dataset"

'''
creates and trains distilled model
Input: folder to output labeled data to.
'''    
def create_base_model(DATASET_DIR_PATH=DATASET_DIR_PATH):
    base_model = GroundingDINO(define_ontology())
    #train base_model on data in /images
    dataset = base_model.label(
        input_folder=IMAGE_DIR_PATH,
        extension=".jpg", #ENSURE PROPER image ext.
        output_folder=DATASET_DIR_PATH)
    return base_model

'''
Define ontology
passed to create_base_model
'''
def define_ontology():
    ontology=CaptionOntology({
    "Yeild Sign": "yield sign"
    })
    return ontology

'''
Ruturns autolabeled dataset 
returns: sv.DetectionDataset
'''
def get_autolabeled_dataset(SAVE_TO_FILE = True):
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=IMAGES_DIRECTORY_PATH,
        annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
        data_yaml_path=DATA_YAML_PATH)
    return dataset

'''
Trains distilled model at .pt location
input: path to .yaml of dataset
'''
def train_distilled_model(dataset_path = DATA_YAML_PATH):
    target_model = YOLOv8("yolov8n.pt")
    target_model.train(dataset_path, epochs=50) 
    return target_model

'''
Create distilled model, view samples
'''
if __name__ == '__main__':
    #ensure path to data is correct 
    #display_img_sample()

    #INITIALIZE PARENT MODEL
    parent_model = create_base_model()
    dataset = get_autolabeled_dataset()
    display_autolabeld_dataset(dataset)

    #DEFINE CLASSES TO TRAIN
    #define_ontology()
