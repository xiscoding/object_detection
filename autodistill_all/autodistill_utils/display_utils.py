import supervision as sv
import os
import cv2

HOME = "/home/xdoestech/Desktop/object_detection/autodistill_all"
IMAGE_DIR_PATH = f"{HOME}/images"
SAMPLE_SIZE = 16
SAMPLE_GRID_SIZE = (4, 4)
SAMPLE_PLOT_SIZE = (16, 16)

def display_img_sample(IMAGE_DIR_PATH = IMAGE_DIR_PATH,
                        SAMPLE_SIZE = SAMPLE_SIZE,
                        SAMPLE_GRID_SIZE = SAMPLE_GRID_SIZE,
                        SAMPLE_PLOT_SIZE = SAMPLE_PLOT_SIZE
                        ):
    image_paths = sv.list_files_with_extensions(
    directory=IMAGE_DIR_PATH,
    extensions=["png", "jpg", "jpg"])
    print('image count:', len(image_paths)) 
    titles = [
        image_path.stem
        for image_path
        in image_paths[:SAMPLE_SIZE]]
    images = [
        cv2.imread(str(image_path))
        for image_path
        in image_paths[:SAMPLE_SIZE]]
    sv.plot_images_grid(images=images, titles=titles, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE)


def display_autolabeld_dataset(dataset):
    image_names = list(dataset.images.keys())[:SAMPLE_SIZE]

    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()

    images = []
    for image_name in image_names:
        image = dataset.images[image_name]
        annotations = dataset.annotations[image_name]
        labels = [
            dataset.classes[class_id]
            for class_id
            in annotations.class_id]
        annotates_image = mask_annotator.annotate(
            scene=image.copy(),
            detections=annotations)
        annotates_image = box_annotator.annotate(
            scene=annotates_image,
            detections=annotations,
            labels=labels)
        images.append(annotates_image)

    sv.plot_images_grid(
        images=images,
        titles=image_names,
        grid_size=SAMPLE_GRID_SIZE,
        size=SAMPLE_PLOT_SIZE)
    
if __name__ == '__main__':
    display_img_sample(IMAGE_DIR_PATH,
                        SAMPLE_SIZE,
                        SAMPLE_GRID_SIZE,
                        SAMPLE_PLOT_SIZE)
    