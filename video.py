import cv2
import os
import re
from glob import glob

def create_video_from_images(image_folder, output_video_path, fps=5):
    # 使用正則表達式匹配圖片文件名稱中的數字
    image_paths = glob(os.path.join(image_folder, "aggregated_confusion_matrix_epoch_*.png"))
    image_paths = sorted(image_paths, key=lambda x: int(re.search(r'(\d+)', x).group(0)))  # 根據數字排序

    # 確認至少有一張圖片
    if not image_paths:
        print("沒有找到符合的圖片文件")
        return

    # 讀取第一張圖片以獲取寬和高
    frame = cv2.imread(image_paths[0])
    height, width, layers = frame.shape

    # 初始化影片編碼器
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 將圖片寫入影片
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 釋放影片編碼器資源
    video_writer.release()
    print(f"影片已成功保存到 {output_video_path}")

# 使用範例
image_folder = "./inference_results"  # 設置圖片所在的資料夾路徑
output_video_path = "aggregated_confusion_matrix_video.mp4"  # 設置輸出影片文件名
create_video_from_images(image_folder, output_video_path, fps=5)
