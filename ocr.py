import os
import json
import numpy as np
from paddleocr import PaddleOCR
import pyheif
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, source_dir, target_dir):
        self.source_dir = source_dir
        self.target_dir = target_dir

        # 目标文件夹不存在，则创建
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        self.processed_files_path = os.path.join(target_dir, ".processed_files")
        self.count_map_path = os.path.join(target_dir, ".count_map.json")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', drop_score=0, show_log=False,
                             rec_model_dir='/paddle/paddle-ocr/PaddleOCR/ch_PP-OCRv4_rec_server_infer')
        # self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', drop_score=0, show_log=False)

        self.processed_files = self.get_processed_files()
        self.processed_files_handle = open(self.processed_files_path, 'a')

        self.count_map = self.load_count_map()


        # 创建置信度文件夹
        confidence_levels = [str(i) for i in range(0, 100, 10)]
        for level in confidence_levels:
            level_dir = os.path.join(target_dir, level)
            if not os.path.exists(level_dir):
                os.makedirs(level_dir)

        # 创建未处理文件夹
        self.unprocessed_dir = os.path.join(target_dir, 'unprocessed')
        if not os.path.exists(self.unprocessed_dir):
            os.makedirs(self.unprocessed_dir)

    def __del__(self):
        if self.processed_files_handle:
            self.processed_files_handle.close()

    def convert_heic_to_ndarray(self, heic_path):
        heif_file = pyheif.read(heic_path)
        image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride)
        image = image.convert('RGB')  # 确保图像为RGB格式
        # image.save(heic_path + '.jpg', 'JPEG')
        return np.array(image)

    def recognize_text(self, image_ndarray):
        try:
            result = self.ocr.ocr(image_ndarray, cls=True)
            if result and result[0]:
                text, confidence = max([(line[1][0], line[1][1]) for line in result[0]], key=lambda x: x[1])
                return text, confidence
        except Exception as e:
            print(f"OCR识别时出错: {e}")
        return None, None

    def record_processed_file(self, filename):
        self.processed_files_handle.write(f"{filename}\n")
        self.processed_files_handle.flush()

    def get_processed_files(self):
        if not os.path.exists(self.processed_files_path):
            return set()
        with open(self.processed_files_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)

    def load_count_map(self):
        if os.path.exists(self.count_map_path):
            with open(self.count_map_path, 'r', encoding='utf-8') as f:
                return defaultdict(int, json.load(f))
        return defaultdict(int)

    def save_count_map(self):
        with open(self.count_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.count_map, f, ensure_ascii=False, indent=4)

    def process_images(self):
        # 获取源目录中所有HEIC图片
        heic_files = [f for f in os.listdir(self.source_dir) if f.lower().endswith('.heic')]

        total = 0
        # 使用tqdm创建进度条
        for filename in tqdm(heic_files, desc='处理进度'):
            if filename in self.processed_files:
                continue  # 跳过已处理的文件

            heic_path = os.path.join(self.source_dir, filename)

            # 转换格式并获取JPG图像的二进制数据
            img_ndarray = self.convert_heic_to_ndarray(heic_path)

            # 识别图片中的文字
            text, confidence = self.recognize_text(img_ndarray)

            # 如果识别到文字，按要求重命名和移动文件
            if text and confidence:
                self.count_map[text] = self.count_map.get(text, 0) + 1
                new_filename = f"{text}_{self.count_map[text]}.heic"
                confidence_level = str(int(confidence * 100 // 10 * 10))
                target_path = os.path.join(self.target_dir, confidence_level, new_filename)

                # 移动和重命名文件
                os.rename(heic_path, target_path)
            else:
                # 如果无法识别文字，移动到未处理文件夹
                unprocessed_path = os.path.join(self.unprocessed_dir, filename)
                os.rename(heic_path, unprocessed_path)

            # 记录已处理的文件
            self.record_processed_file(filename)
            if total % 10 == 0:
                self.save_count_map()
            total+=1

        if os.path.exists(self.count_map_path):
            os.remove(self.count_map_path)
        if os.path.exists(self.processed_files_path):
            os.remove(self.processed_files_path)
        print('处理完成。')

if __name__ == "__main__":
    # source_dir = '/paddle/dataset/chinese'  # 源目录
    source_dir = '/paddle/dataset/result/unprocessed'  # 源目录
    target_dir = '/paddle/dataset/result'  # 目标目录

    processor = ImageProcessor(source_dir, target_dir)
    processor.process_images()
    # image_ndarray = processor.convert_heic_to_ndarray('/paddle/dataset/result/90/艾_1.heic')
    # text, confidence = processor.recognize_text(image_ndarray)
    # print(text)
    # print(confidence)
