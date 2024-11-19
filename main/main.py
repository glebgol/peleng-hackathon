import cv2
import numpy as np
import pandas as pd
import os

# Параметры
video_path = r"D:\Hackathon\MainVideo\Материалы_Для_ТЗ\step1\videoset1\Seq1_camera1.mov"
output_folder = r"D:\Hackathon\photos"
output_data_file = r"D:\Hackathon\data.txt"
template_path = r"D:\Hackathon\Sphera\Sph8.png"

# Создаем выходную папку, если она не существует
os.makedirs(output_folder, exist_ok=True)

# Загружаем изображение объекта
template = cv2.imread(template_path)
if template is None:
    raise ValueError("Не удалось загрузить шаблон изображения. Проверьте путь.")
template_height, template_width = template.shape[:2]

# Открываем видео
cap = cv2.VideoCapture(video_path)

# Инициализация списка для записи данных
data = []
frame_number = 0

# Обработка видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Применяем шаблонное соответствие
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7  # Минимальная точность
    yloc, xloc = np.where(result >= threshold)

    # Убираем дубликаты координат (если совпадения перекрываются)
    points = list(zip(xloc, yloc))
    points = list(set(points))

    # Рисуем прямоугольники вокруг найденных объектов
    for (x, y) in points:
        cv2.rectangle(frame, (x, y), (x + template_width, y + template_height), (0, 255, 0), 2)

        # Сохраняем координаты объекта
        object_x = x + template_width // 2
        object_y = y + template_height // 2
        object_length = (template_width + template_height) // 2  # Пример длины объекта

        # Сохраняем данные
        data.append({
            'Frame': frame_number,
            'Center_X': object_x,
            'Center_Y': object_y,
            'Length': object_length
        })

    # Отображаем кадр
    cv2.imshow('Tracking', frame)

    # Ждем 1 мс и проверяем, была ли нажата клавиша 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()

# Сохраняем данные в текстовый файл
df = pd.DataFrame(data)
df.to_csv(output_data_file, index=False, sep='\t')

print("Обработка завершена! Данные сохранены.")