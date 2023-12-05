# Local
import cv2
import numpy as np


def bernsen_thresholding(image, e=15, r=15):
    # Копирование изображения для обработки
    processed_image = np.copy(image)

    # Итерация по каждому квадрату
    half_r = r // 2
    for x in range(half_r, image.shape[1] - half_r):
        for y in range(half_r, image.shape[0] - half_r):
            # Получение яркостей пикселей в пределах квадрата
            pixel_values = []
            for i in range(x - half_r, x + half_r + 1):
                for j in range(y - half_r, y + half_r + 1):
                    pixel_values.append(image[j, i])

            # Вычисление наименьшего и наибольшего уровня яркости
            jlow = np.min(pixel_values)
            jhigh = np.max(pixel_values)

            # Вычисление порога
            threshold = (jhigh - jlow) / 2

            # Пороговая обработка пикселя
            if threshold <= e:
                processed_image[y, x] = 0  # Замена пикселя на черный
            else:
                processed_image[y, x] = 255  # Замена пикселя на белый

    return processed_image


def niblack_thresholding(image, r=15, k=-0.2):
    # Копирование изображения для обработки
    processed_image = np.copy(image)

    # Итерация по каждому пикселю
    half_r = r // 2
    for x in range(half_r, image.shape[1] - half_r):
        for y in range(half_r, image.shape[0] - half_r):
            # Получение локальной окрестности
            neighborhood = image[y - half_r:y + half_r + 1, x - half_r:x + half_r + 1]

            # Вычисление среднего и среднеквадратического отклонения
            mean_value = np.mean(neighborhood)
            std_deviation = np.std(neighborhood)

            # Вычисление порога
            threshold = mean_value + k * std_deviation

            # Пороговая обработка пикселя
            if image[y, x] <= threshold:
                processed_image[y, x] = 0  # Замена пикселя на черный
            else:
                processed_image[y, x] = 255  # Замена пикселя на белый

    return processed_image


def adaptive_thresholding(image, constant=1 / 3, block_size=11):
    # Преобразование изображения в оттенки серого

    # Применение адаптивного порогового преобразования
    processed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size,
                                      1 / constant)

    return processed


def global_thresholding_by_histogramm(image):
    # Вычисление гистограммы изображения
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Вычисление общего числа пикселей в изображении
    total_pixels = image.shape[0] * image.shape[1]

    # Расчет накопленной суммы гистограммы
    cumulative_sum = np.cumsum(hist)

    # Нахождение порога на основе метода Оцу
    threshold = 0
    max_variance = 0

    for t in range(256):
        # Вычисление вероятностей классов
        w0 = cumulative_sum[t] / total_pixels
        w1 = (total_pixels - cumulative_sum[t]) / total_pixels

        # Вычисление средних значений классов
        mu0 = np.sum(np.multiply(hist[:t + 1], np.arange(t + 1))) / (cumulative_sum[t] + 1e-8)
        mu1 = np.sum(np.multiply(hist[t + 1:], np.arange(t + 1, 256))) / (
                cumulative_sum[255] - cumulative_sum[t] + 1e-8)

        # Вычисление межклассовой дисперсии
        variance = w0 * w1 * (mu0 - mu1) ** 2

        # Обновление порога, если дисперсия больше максимальной
        if variance > max_variance:
            max_variance = variance
            threshold = t

    # Применение порогового преобразования
    processed = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]

    return processed


def global_thresholding_otsu(image):
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded


def logarithmic_transformation(image):
    # Вычисление максимального значения яркости в изображении
    max_value = np.max(image)

    # Применение логарифмического преобразования
    transformed = 255 * (np.log1p(image) / np.log1p(max_value))

    # Округление значений и приведение к целочисленному типу
    transformed = np.round(transformed).astype(np.uint8)

    return transformed


def linear_contrast(image):
    # Вычисление минимального и максимального значения яркости в изображении
    min_value = np.min(image)
    max_value = np.max(image)

    # Применение линейного контрастирования
    transformed = 255 * ((image - min_value) / (max_value - min_value))
    # Округление значений и приведение к целочисленному типу
    transformed = np.round(transformed).astype(np.uint8)

    return transformed


def median_filter(image, kernel_size=3):
    # Применение медианного фильтра
    filtered_image = cv2.medianBlur(image, kernel_size)

    return filtered_image


def minimum_filter(image, kernel_size=3):
    # Размер изображения
    height, width = image.shape

    # Половина размера окна
    k = kernel_size // 2

    # Создание выходного изображения
    filtered_image = np.zeros_like(image)

    # Применение фильтра минимума
    for i in range(k, height - k):
        for j in range(k, width - k):
            # Выделение окрестности
            neighborhood = image[i - k: i + k + 1, j - k: j + k + 1]

            # Замена значения пикселя на минимальное значение в окрестности
            filtered_image[i, j] = np.min(neighborhood)

    return filtered_image


def maximum_filter(image, kernel_size=3):
    # Размер изображения
    height, width = image.shape

    # Половина размера окна
    k = kernel_size // 2

    # Создание выходного изображения
    filtered_image = np.zeros_like(image)

    # Применение фильтра максимума
    for i in range(k, height - k):
        for j in range(k, width - k):
            # Выделение окрестности
            neighborhood = image[i - k: i + k + 1, j - k: j + k + 1]

            # Замена значения пикселя на максимальное значение в окрестности
            filtered_image[i, j] = np.max(neighborhood)

    return filtered_image


def laplacian_filter(image):
    # Определение ядра фильтра Лапласа
    kernel = np.array(
        [[-1, -1, -1],
         [-1, 9, -1],
         [-1, -1, -1]], dtype=np.float32)

    # Применение свертки с ядром фильтра Лапласа
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image


# # Загрузка изображения с помощью OpenCV
# image = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
#
# # Применение адаптивной пороговой обработки
# processed_image = laplacian_filter(image)
#
# # Сохранение обработанного изображения с помощью OpenCV
# cv2.imwrite('processed_image_laplacian.png', processed_image)
import tkinter as tk
from tkinter import filedialog
import os
import cv2

def select_input_directory():
    input_directory = filedialog.askdirectory(title="Select Input Directory")
    input_entry.delete(0, tk.END)
    input_entry.insert(0, input_directory)

def select_output_directory():
    output_directory = filedialog.askdirectory(title="Select Output Directory")
    output_entry.delete(0, tk.END)
    output_entry.insert(0, output_directory)

def process_images():
    input_directory = input_entry.get()
    output_directory = output_entry.get()

    methods = ['method1', 'method2', 'method3']  # Замените на фактические методы обработки

    for method in methods:
        method_output_directory = os.path.join(output_directory, method)
        os.makedirs(method_output_directory, exist_ok=True)

        image_files = os.listdir(input_directory)
        for image_file in image_files:
            if image_file.endswith('.png'):
                image_path = os.path.join(input_directory, image_file)
                image = cv2.imread(image_path)

                # Процесс обработки изображения с использованием выбранного метода
                processed_image = process_image(image, method)  # Замените process_image на фактическую функцию обработки

                output_image_path = os.path.join(method_output_directory, image_file)
                cv2.imwrite(output_image_path, processed_image)

    print("Processing complete!")

# Функция для обработки изображения с использованием выбранного метода
def process_image(image, method):
    # Реализуйте свою логику обработки изображения для каждого метода
    processed_image = image  # Здесь просто возвращаем исходное изображение
    return processed_image

# Создание графического интерфейса
root = tk.Tk()

# Кнопка 1 - выбор директории с исходными изображениями в формате PNG
button1 = tk.Button(root, text="Select Input Directory", command=select_input_directory)
button1.pack()

# Поле ввода для отображения выбранной директории
input_entry = tk.Entry(root)
input_entry.pack()

# Кнопка 2 - выбор директории для сохранения обработанных изображений
button2 = tk.Button(root, text="Select Output Directory", command=select_output_directory)
button2.pack()

# Поле ввода для отображения выбранной директории
output_entry = tk.Entry(root)
output_entry.pack()

# Кнопка 3 - обработка изображений
button3 = tk.Button(root, text="Process Images", command=process_images)
button3.pack()

root.mainloop()

