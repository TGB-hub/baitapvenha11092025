# baitapvenha11092025
# =============================================================================
# BƯỚC 1: CÀI ĐẶT THƯ VIỆN
# =============================================================================
!pip install requests pillow opencv-python scikit-learn tensorflow matplotlib numpy seaborn

import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
from google.colab import files
import pickle
import random

print("✅ Đã cài đặt thành công tất cả thư viện!")
print(f"TensorFlow version: {tf.__version__}")

# =============================================================================
# BƯỚC 2: CẤU HÌNH
# =============================================================================
FOOD_CLASSES = ['Phở', 'Bún Bò', 'Bánh Xèo', 'Cơm Tấm', 'Bánh Mì']
IMAGE_SIZE = (60, 60)
NUM_IMAGES_PER_CLASS = 50
BATCH_SIZE = 32
EPOCHS = 50

DATASET_PATH = 'dataset'
MODEL_PATH = 'food_classifier_model.h5'
DATA_PROCESSED_PATH = 'processed_data.pkl'

os.makedirs(DATASET_PATH, exist_ok=True)
for food_class in FOOD_CLASSES:
    os.makedirs(f'{DATASET_PATH}/{food_class}', exist_ok=True)

print(f"🎯 Sẽ phân loại {len(FOOD_CLASSES)} loại món ăn:")
for i, food in enumerate(FOOD_CLASSES):
    print(f"   {i+1}. {food}")

# =============================================================================
# BƯỚC 3: KIỂM TRA MÔ HÌNH
# =============================================================================
def check_existing_model():
    model_exists = os.path.exists(MODEL_PATH)
    data_exists = os.path.exists(DATA_PROCESSED_PATH)

    print("🔍 Kiểm tra mô hình đã học:")
    print(f"   📁 Mô hình AI: {'✅ Có' if model_exists else '❌ Chưa có'}")
    print(f"   📁 Dữ liệu xử lý: {'✅ Có' if data_exists else '❌ Chưa có'}")

    if model_exists and data_exists:
        print("🎉 Đã tìm thấy mô hình đã học! Sẽ load mô hình cũ.")
        return True
    else:
        print("🔄 Chưa có mô hình, sẽ bắt đầu học từ đầu.")
        return False

# =============================================================================
# BƯỚC 4: TẠO DỮ LIỆU ẢNH GIẢ LẬP (NÉT HƠN)
# =============================================================================
def generate_synthetic_food_image_bw(food_type, image_index, size=(60, 60)):
    gray_patterns = {
        'Phở': {'base_gray': 120, 'accent_grays': [200, 80, 160, 40]},
        'Bún Bò': {'base_gray': 80, 'accent_grays': [180, 60, 140, 30]},
        'Bánh Xèo': {'base_gray': 200, 'accent_grays': [100, 150, 70, 220]},
        'Cơm Tấm': {'base_gray': 240, 'accent_grays': [120, 180, 90, 60]},
        'Bánh Mì': {'base_gray': 160, 'accent_grays': [100, 200, 140, 80]}
    }

    img = np.ones((size[0], size[1]), dtype=np.uint8)
    pattern = gray_patterns[food_type]
    base_gray = pattern['base_gray']

    # Base + noise (giảm noise)
    for i in range(size[0]):
        for j in range(size[1]):
            noise = np.random.randint(-10, 10)
            gray_value = np.clip(base_gray + noise, 0, 255)
            img[i, j] = gray_value

    # Accent details (nhiều chi tiết hơn)
    accent_grays = pattern['accent_grays']
    num_details = random.randint(10, 25)
    for _ in range(num_details):
        x = random.randint(2, size[0]-8)
        y = random.randint(2, size[1]-8)
        w = random.randint(3, 7)
        h = random.randint(3, 7)
        gray_value = random.choice(accent_grays)
        img[x:x+w, y:y+h] = gray_value

    # Texture
    if random.random() > 0.3:
        if food_type in ['Phở', 'Bún Bò']:
            for _ in range(random.randint(6, 12)):
                start_y = random.randint(0, size[1])
                end_y = random.randint(0, size[1])
                x_pos = random.randint(0, size[0]-1)
                gray_line = random.choice(accent_grays)
                cv2.line(img, (start_y, x_pos), (end_y, x_pos), gray_line, thickness=1)
        elif food_type in ['Bánh Xèo', 'Cơm Tấm']:
            for _ in range(random.randint(4, 9)):
                center_x = random.randint(10, size[0]-10)
                center_y = random.randint(10, size[1]-10)
                radius = random.randint(2, 6)
                gray_circle = random.choice(accent_grays)
                cv2.circle(img, (center_y, center_x), radius, gray_circle, -1)

    # Sharpen
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel_sharpen)

    return img

def collect_training_data():
    print("🔄 Bắt đầu thu thập dữ liệu huấn luyện...")
    total_images = 0
    for food_class in FOOD_CLASSES:
        for img_idx in range(NUM_IMAGES_PER_CLASS):
            synthetic_img_bw = generate_synthetic_food_image_bw(food_class, img_idx)
            img_path = f'{DATASET_PATH}/{food_class}/img_{img_idx:03d}.jpg'
            cv2.imwrite(img_path, synthetic_img_bw)
            total_images += 1
    print(f"🎉 Đã tạo {total_images} ảnh ĐEN TRẮNG (nét hơn)")
    return total_images

# =============================================================================
# BƯỚC 5: XỬ LÝ DỮ LIỆU
# =============================================================================
def load_and_preprocess_data():
    X, y = [], []
    for class_idx, food_class in enumerate(FOOD_CLASSES):
        class_path = f'{DATASET_PATH}/{food_class}'
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            img_bw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            img_bw = cv2.GaussianBlur(img_bw, (3, 3), 0)
            img_bw = cv2.equalizeHist(img_bw)

            img_normalized = img_bw.astype(np.float32) / 255.0
            X.append(img_normalized)
            y.append(class_idx)
    X, y = np.array(X), np.array(y)
    with open(DATA_PROCESSED_PATH, 'wb') as f:
        pickle.dump((X, y), f)
    return X, y

def load_processed_data():
    with open(DATA_PROCESSED_PATH, 'rb') as f:
        X, y = pickle.load(f)
    return X, y

# =============================================================================
# BƯỚC 6: MÔ HÌNH CNN
# =============================================================================
def create_cnn_model():
    model = keras.Sequential([
        layers.Input(shape=(60, 60, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(FOOD_CLASSES), activation='softmax')
    ])
    return model

# =============================================================================
# BƯỚC 7: HUẤN LUYỆN HOẶC LOAD
# =============================================================================
has_existing_model = check_existing_model()

if has_existing_model:
    model = keras.models.load_model(MODEL_PATH)
    X, y = load_processed_data()
else:
    collect_training_data()
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = create_cnn_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X_train_cnn = X_train.reshape(-1, 60, 60, 1)
    X_test_cnn = X_test.reshape(-1, 60, 60, 1)
    history = model.fit(X_train_cnn, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test_cnn, y_test), verbose=1)
    model.save(MODEL_PATH)

# =============================================================================
# BƯỚC 8: HÀM KIỂM TRA ẢNH
# =============================================================================
def preprocess_test_image_bw(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không thể đọc ảnh!")

    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_gray = cv2.equalizeHist(img_gray)

    img_resized = cv2.resize(img_gray, IMAGE_SIZE)

    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    img_resized = cv2.filter2D(img_resized, -1, kernel_sharpen)

    img_normalized = img_resized.astype(np.float32) / 255.0

    return img, img_gray, img_resized, img_normalized

def predict_food_image_bw(image_path, show_result=True):
    try:
        img_original, img_gray, img_resized, img_processed = preprocess_test_image_bw(image_path)
        img_input = img_processed.reshape(1, 60, 60, 1)
        predictions = model.predict(img_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100
        predicted_food = FOOD_CLASSES[predicted_class_idx]

        if show_result:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            if len(img_original.shape) == 3:
                img_display = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
                axes[0].imshow(img_display)
            else:
                axes[0].imshow(img_original, cmap='gray')
            axes[0].set_title('🖼️ Ảnh gốc (full size)', fontsize=14)
            axes[0].axis('off')

            axes[1].imshow(img_gray, cmap='gray')
            axes[1].set_title('⚫ Ảnh ĐEN TRẮNG (tăng nét)', fontsize=14)
            axes[1].axis('off')

            axes[2].imshow(img_resized, cmap='gray')
            axes[2].set_title('📐 60x60 (input AI)', fontsize=14)
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()

            print("="*50)
            print("🎯 KẾT QUẢ DỰ ĐOÁN")
            print("="*50)
            print(f"🍜 Món ăn được dự đoán: {predicted_food}")
            print(f"📊 Độ tin cậy: {confidence:.2f}%")
            print("="*50)
            print("\n📈 Chi tiết độ tin cậy cho từng món:")
            for i, food in enumerate(FOOD_CLASSES):
                prob = predictions[0][i] * 100
                bar = "█" * int(prob // 2)
                print(f"{food:12}: {prob:6.2f}% {bar}")

        return predicted_food, confidence, predictions[0]
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        return None, 0, None

# =============================================================================
# BƯỚC 9: UPLOAD ẢNH TEST
# =============================================================================
def test_with_uploaded_image():
    uploaded = files.upload()
    for filename, data in uploaded.items():
        with open(filename, 'wb') as f:
            f.write(data)
        predict_food_image_bw(filename)
        os.remove(filename)

# =============================================================================
# BƯỚC 10: TEST ẢNH DO NGƯỜI DÙNG CHỌN
# =============================================================================
print("\n🧪 Vui lòng upload ảnh cần test:")
test_with_uploaded_image()

print("\n🎉 PHẦN MỀM HOÀN THÀNH (ẢNH NÉT HƠN)!")
