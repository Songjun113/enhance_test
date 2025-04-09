import cv2
import numpy as np
import os
from scipy.ndimage import distance_transform_edt

def compute_difference(label_img: np.ndarray, test_img: np.ndarray) -> float:
    """
    给定一张label图像与一张测试图像，计算difference值，即肿瘤区域内部与外部边缘灰度均值之差的绝对值。
    """
    # 1. 二值化label图像
    _, label_bin = cv2.threshold(label_img, 127, 255, cv2.THRESH_BINARY)

    # 2. 查找轮廓，选择最大目标轮廓
    contours, _ = cv2.findContours(label_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("在label图像中未找到任何轮廓。")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    # 3. 构建目标区域掩模
    mask = np.zeros_like(label_bin)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
    num = np.count_nonzero(mask == 255)

    # 4. 提取并计算内部区域的灰度均值
    result = test_img[mask == 255].mean()

    # 5. 构建边缘掩模
    edge_mask = np.zeros_like(label_bin)
    cv2.drawContours(edge_mask, [largest_contour], -1, 255, thickness=1)

    # 6. 距离变换（反转边缘图像后计算）
    edge_binary = (edge_mask == 255).astype(np.uint8)
    distance_map = distance_transform_edt(1 - edge_binary)

    # 7. 避免选取内部区域
    distance_map[mask == 255] = np.inf

    # 8. 获取外部区域的坐标与距离
    y, x = np.where(mask == 0)
    distances = distance_map[y, x]

    # 9. 选择距离边缘最近的 num 个外部像素
    sorted_indices = np.argsort(distances)
    if len(sorted_indices) < num:
        raise ValueError("外部区域像素数量不足，无法匹配内部区域。")
    selected_y = y[sorted_indices[:num]]
    selected_x = x[sorted_indices[:num]]

    # 10. 计算外部区域灰度均值
    answer = test_img[selected_y, selected_x].mean()

    # 11. 返回 difference
    return abs(result - answer)

# 输入路径设定
raw_pic_folder = "raw_pic"
my_label_folder = "my_label"
processed_pic_folder = "processed_pic"

# 加载 my_label 图像
label_filenames = os.listdir(my_label_folder)
if not label_filenames:
    raise FileNotFoundError("my_label 文件夹为空。")
label_path = os.path.join(my_label_folder, label_filenames[0])
label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
if label_img is None:
    raise ValueError(f"无法读取label图像: {label_path}")

# 加载 raw_pic 图像
raw_filenames = os.listdir(raw_pic_folder)
if not raw_filenames:
    raise FileNotFoundError("raw_pic 文件夹为空。")
raw_path = os.path.join(raw_pic_folder, raw_filenames[0])
raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
if raw_img is None:
    raise ValueError(f"无法读取原始图像: {raw_path}")

# 计算 my_label 和原始图像的 difference
raw_difference = compute_difference(label_img, raw_img)
print(f"\n原始图像与标签的difference值为: {raw_difference:.2f}\n")

# 遍历 processed_pic 文件夹中的所有图像
processed_filenames = sorted(os.listdir(processed_pic_folder))
if not processed_filenames:
    raise FileNotFoundError("processed_pic 文件夹为空。")

# 存储每个sjj_answer结果
for fname in processed_filenames:
    test_path = os.path.join(processed_pic_folder, fname)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if test_img is None:
        print(f"跳过无法读取的图像: {fname}")
        continue
    try:
        test_difference = compute_difference(label_img, test_img)
        sjj_answer = abs(test_difference - raw_difference)
        print(f"增强图像 {fname} 的difference值为: {test_difference:.2f}，sjj_answer为: {sjj_answer:.2f}")
    except Exception as e:
        print(f"处理图像 {fname} 时出错: {e}")
