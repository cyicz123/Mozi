import logging
import os
import json
import shutil
from paddleocr import PaddleOCR
import pyheif
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import paddle
import threading
from opencc import OpenCC

class TraditionalToSimplified:
    def __init__(self):
        self.cc = OpenCC('t2s')

    def Convert(self, text):
        return self.cc.convert(text)

class ImageConverter:
    def __init__(self, src_dir, dst_dir, num_threads=4) -> None:
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.logger = logging.getLogger('ImageConverter')
        self.logger.setLevel(logging.DEBUG)
        self.num_threads = num_threads
    
    def __call__(self):
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)
        
        files = os.listdir(self.src_dir)
        files_per_thread = len(files) // self.num_threads

        threads = []
        for i in range(self.num_threads):
            start = i * files_per_thread
            end = (i + 1) * files_per_thread if i < self.num_threads - 1 else len(files)
            t = threading.Thread(target=self._convert_thread, args=(files[start:end],))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        
    def _convert_thread(self, files):
        for file in files:
            image_format_endwith = ('.png', '.heic', '.heif', '.jpg', '.jpeg', 'webp', 'tif', 'bmp')
            src_file = os.path.join(self.src_dir, file)
            dst_file = os.path.join(self.dst_dir, file.replace(file.split('.')[-1], 'jpg'))
            if os.path.exists(dst_file):
                continue
            if file.lower().endswith(image_format_endwith):
                if file.lower().endswith('.heic'):
                    heif = pyheif.read(src_file)
                    image = Image.frombytes(
                        heif.mode, heif.size, heif.data,
                        "raw", heif.mode, heif.stride,
                    )
                    image = image.convert('RGB')  # 确保图像为RGB格式
                    image.save(dst_file, 'JPEG')
                else:
                    shutil.copy(src_file, dst_file)
            else:
                self.logger.error(f"Unsupported file type: {file}")

class FileNameCountMap:
    def __init__(self, dir):
        self.dir = dir
        self.count_map_path = os.path.join(dir, ".count_map.json")
        self.count_map = self.load_count_map()
        self.load_index = 0
        
    def load_count_map(self):
        if os.path.exists(self.count_map_path):
            with open(self.count_map_path, 'r', encoding='utf-8') as f:
                return defaultdict(int, json.load(f))
        return defaultdict(int)
    
    def delete_count_map(self):
        if os.path.exists(self.count_map_path):
            os.remove(self.count_map_path)

    def save_count_map(self):
        with open(self.count_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.count_map, f, ensure_ascii=False, indent=4, sort_keys=True)
    
    def GetValue(self, key):
        self.count_map[key] = self.count_map.get(key, 0) + 1
        if self.load_index % 10 == 9:
            self.save_count_map()
            self.load_index += 1
        return self.count_map[key]


class ImageProcessor:
    def __init__(self, target_dir):
        self.target_dir = target_dir

        # 目标文件夹不存在，则创建
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if paddle.is_compiled_with_cuda():
            # 尝试将设备设置为GPU
            print("GPU is available, using GPU")
            paddle.set_device('gpu')
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', drop_score=0, show_log=False,
                                det=False, 
                                use_gpu=True)
        else:
            print("GPU is not available, using CPU")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', drop_score=0, show_log=False,
                                det=False, 
                                use_gpu=False)
        
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

        # 加载已处理文件列表
        self.file_count_map = FileNameCountMap(target_dir)
        # 创建繁体转换类
        self.traditional_to_simp = TraditionalToSimplified()

    def recognize_text(self, image_file):
        try:
            result = self.ocr.ocr(image_file)
            if result and result[0]:
                text, confidence = result[0][0][1]
                return text, confidence
        except Exception as e:
            print(f"OCR识别时出错: {e}")
        return None, None


    def process_images(self):
        # 使用tqdm创建进度条
        for filename in tqdm(os.listdir(self.target_dir), desc='Processing progress'):
            if not filename.endswith('.jpg'):
                continue

            img_path = os.path.join(self.target_dir, filename)

            # 识别图片中的文字
            text, confidence = self.recognize_text(img_path)

            # 如果识别到文字，按要求重命名和移动文件
            if text and confidence:
                text = self.traditional_to_simp.Convert(text)
                new_filename = f"{text}_{self.file_count_map.GetValue(text)}.jpg"
                confidence_level = str(int(confidence * 100 // 10 * 10))
                target_path = os.path.join(self.target_dir, confidence_level, new_filename)

                # 移动和重命名文件
                os.rename(img_path, target_path)
            else:
                # 如果无法识别文字，移动到未处理文件夹
                unprocessed_path = os.path.join(self.unprocessed_dir, filename)
                os.rename(img_path, unprocessed_path)

        self.file_count_map.delete_count_map()
        print('处理完成。')

if __name__ == "__main__":
    source_dir = 'data/chinese'  # 源目录
    target_dir = 'data/result'  # 目标目录

    converter  = ImageConverter(source_dir, target_dir)
    converter()
    processor = ImageProcessor(target_dir)
    processor.process_images()
    # text, confidence = processor.recognize_text(image_ndarray)
    # image = Image.fromarray(image_ndarray)
    # image.save('test.jpg', 'JPEG')
    # print(text)
    # print(confidence)
