import numpy as np
import pandas as pd


# Функция для проверки выбросов
def is_outlier(new_point, last_point, threshold):
    if last_point is None:
        return False
    dist = np.sqrt((new_point[0] - last_point[0]) ** 2 + (new_point[1] - last_point[1]) ** 2)
    return dist > threshold

def interpolate_track_points(points):
    # Удаляем точки с None
    valid_points = [(x, y) if x is not None and y is not None else (np.nan, np.nan) for x, y in points]
    
    # Преобразуем список в DataFrame для интерполяции
    df_points = pd.DataFrame(valid_points, columns=['x', 'y'])
    df_points = df_points.interpolate()  # Интерполируем пропущенные значения
    df_points = df_points.bfill()  # Заполняем значения до интерполяции, если начало данных отсутствует
    df_points = df_points.ffill()  # Заполняем значения после интерполяции, если есть пропуски в конце
    return df_points.to_numpy().tolist()

