import os
from paddleocr import PaddleOCR
import pyheif
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

# 初始化PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch', drop_score=0)

# 转换HEIC到JPG格式
def convert_heic_to_jpg(heic_path, jpg_path):
    heif_file = pyheif.read(heic_path)
    image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride)
    image.save(jpg_path, "JPEG")

# OCR识别并返回结果和置信度
def recognize_text(image_path):
  try:
    result = ocr.ocr(image_path, cls=True)
    if result:
        # 获取最高置信度的文本和其置信度
        text, confidence = max([(line[1][0], line[1][1]) for line in result[0]], key=lambda x: x[1])
        return text, confidence
  except Exception as e:
    print(f"OCR识别时出错: {e}")
  return None, None

# 记录已处理的文件
def record_processed_file(filename, processed_files_path):
    with open(processed_files_path, 'a') as f:
        f.write(f"{filename}\n")

# 读取已处理的文件列表
def get_processed_files(processed_files_path):
    if not os.path.exists(processed_files_path):
        return set()
    with open(processed_files_path, 'r') as f:
        return set(line.strip() for line in f)

# 主程序
source_dir = '/paddle/dataset/test'  # 源目录
target_dir = '/paddle/dataset/test'  # 目标目录
processed_files_path = os.path.join(source_dir, ".processed_files")
count_map = defaultdict(int)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 创建置信度文件夹
confidence_levels = [str(i) for i in range(0, 100, 10)]
for level in confidence_levels:
    level_dir = os.path.join(target_dir, level + 's')
    if not os.path.exists(level_dir):
        os.makedirs(level_dir)

# 创建未处理文件夹
unprocessed_dir = os.path.join(target_dir, 'unprocessed')
if not os.path.exists(unprocessed_dir):
    os.makedirs(unprocessed_dir)

heic_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.heic')]
processed_files = get_processed_files(processed_files_path)
# 遍历源目录中的HEIC图片
for filename in tqdm(heic_files, desc='处理进度'):
    if filename in processed_files:
        continue  # 跳过已处理的文件

    if filename.lower().endswith('.heic'):
        heic_path = os.path.join(source_dir, filename)
        jpg_path = heic_path.replace('.HEIC', '.jpg')

        # 转换格式
        convert_heic_to_jpg(heic_path, jpg_path)

        # 识别图片中的文字
        text, confidence = recognize_text(jpg_path)

        # 如果识别到文字，按要求重命名和移动文件
        if text and confidence:
            count_map[text] += 1
            new_filename = f"{text}_{count_map[text]}.heic"
            confidence_level = str(int(confidence * 100 // 10 * 10)) + 's'
            target_path = os.path.join(target_dir, confidence_level, new_filename)

            # 移动和重命名文件
            os.rename(heic_path, target_path)
        else:
            # 如果无法识别文字，移动到未处理文件夹
            unprocessed_path = os.path.join(unprocessed_dir, filename)
            os.rename(heic_path, unprocessed_path)

        # 删除临时jpg文件
        os.remove(jpg_path)
        # 记录已处理的文件
        record_processed_file(filename, processed_files_path)


print('处理完成。')