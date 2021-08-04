from deconfuse import deconfuse
from paddleocr import PaddleOCR, draw_ocr
import os
import cv2
from tqdm import tqdm

if __name__ == "__main__":

    test_images_dir = os.path.join("test_images", "test_videos")
    results_dir = os.path.join("test_images", "test_videos_results_x4")
    ocr = PaddleOCR(lang='en',cls=True)
    scale_percent = 400  # percent of original size

    for image_basename in tqdm(os.listdir(test_images_dir)):

        img_path = os.path.join(test_images_dir, image_basename)
        result = ocr.ocr(img_path,scale_percent = 800)
        for line in result:
            print(line)

        # draw result
        img = cv2.imread(img_path)

        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        plate_strings : [str] = deconfuse(boxes,txts)

        im_show = draw_ocr(img, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
        cv2.imwrite(os.path.join(results_dir, image_basename),im_show)

