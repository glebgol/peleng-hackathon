import cv2
import numpy as np
import pandas as pd
import os
import point_process

folder_path = "photos1"
output_data_file = "data.txt"
template_path = r"Sphera\Sph8.png"

# Загружаем изображение объекта
template = cv2.imread(template_path)
if template is None:
    raise ValueError("Не удалось загрузить шаблон изображения. Проверьте путь.")
template_height, template_width = template.shape[:2]

image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

if not image_files:
    print(f"No image files found in {folder_path}.")
    raise ValueError("No image files found")

# Initialize a list for storing data (if needed)
data = []
track_points = []

# Обработка видео
for frame_number, image_file in enumerate(image_files):
    frame_path = os.path.join(folder_path, image_file)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Error reading image: {frame_path}")
        continue

    # Применяем шаблонное соответствие
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7  # Минимальная точность
    yloc, xloc = np.where(result >= threshold)

    points = list(zip(xloc, yloc))
    values = [result[y, x] for x, y in points] 
    if points:  # Check if points list is not empty
        max_index = np.argmax(values)  # Index of the maximum value
        best_point = points[max_index]
        x,y = best_point
        
        # Сохраняем координаты объекта
        object_x = x + template_width // 2
        object_y = y + template_height // 2
        object_length = (template_width + template_height) // 2  # Пример длины объекта

        # Проверяем, если координаты сильно отличаются от предыдущих, то это выброс
        if track_points and point_process.is_outlier((object_x, object_y), track_points[-1], 500):
            print("Обнаружен выброс, интерполяция...")
            track_points.append((np.nan, np.nan))  # Добавляем NaN для последующей интерполяции
        else:
            # Добавляем в список точек для трека
            track_points.append((object_x, object_y))

        track_points = point_process.interpolate_track_points(track_points)
        data.append({
            'Frame': frame_number,
            'Center_X': (object_x,object_y),
            'Length': object_length
        })
        last_point = track_points[-1]
        x, y = last_point[0], last_point[1]
        x = (int) (x - template_width // 2)
        y = (int) (y - template_width // 2)
        cv2.rectangle(frame, (x, y), (x + template_width, y + template_height), (0, 255, 0), 2)

    # Отображаем кадр
    cv2.imshow('Tracking', frame)

    # Ждем 1 мс и проверяем, была ли нажата клавиша 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cv2.destroyAllWindows()

# Сохраняем данные в текстовый файл
df = pd.DataFrame(data)
df.to_csv(output_data_file, index=False, sep='\t')

print("Обработка завершена! Данные сохранены.")