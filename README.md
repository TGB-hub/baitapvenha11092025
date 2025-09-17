# baitapvenha11092025
# Cài đặt thư viện (bỏ comment nếu chưa cài)
# pip install opencv-python scikit-learn tensorflow matplotlib numpy pillow tkinter

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import pickle
import time
import random
import tkinter as tk
from tkinter import filedialog, messagebox

print(" Đã import thành công tất cả thư viện!")
print(f"TensorFlow version: {tf.__version__}")


def select_image_file():
    """
    Mở hộp thoại để chọn file ảnh từ máy tính
    """
    # Tạo cửa sổ root ẩn
    root = tk.Tk()
    root.withdraw()
    
    # Hiển thị hộp thoại chọn file
    file_path = filedialog.askopenfilename(
        title=" Chọn ảnh để phân loại món ăn",
        filetypes=[
            ("Tất cả ảnh", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("PNG files", "*.png"),
            ("BMP files", "*.bmp"),
            ("GIF files", "*.gif"),
            ("TIFF files", "*.tiff"),
            ("Tất cả files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def show_file_info(file_path):
    """
    Hiển thị thông tin file ảnh
    """
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f" File được chọn: {os.path.basename(file_path)}")
        print(f" Đường dẫn: {file_path}")
        print(f" Kích thước file: {file_size:.1f} KB")
        
        # Đọc ảnh để lấy thông tin
        img = cv2.imread(file_path)
        if img is not None:
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            print(f" Kích thước ảnh: {width}x{height}")
            print(f" Số kênh màu: {channels}")
        else:
            print(" Không thể đọc được ảnh!")
        print("-" * 50)


# Danh sách các món ăn cần phân loại
FOOD_CLASSES = ['Phở', 'Bún Bò', 'Bánh Xèo', 'Cơm Tấm', 'Bánh Mì']
IMAGE_SIZE = (60, 60)  # Kích thước ảnh sau khi resize
NUM_IMAGES_PER_CLASS = 50  # Số ảnh cho mỗi lớp
BATCH_SIZE = 32
EPOCHS = 50

# Đường dẫn lưu trữ
DATASET_PATH = 'dataset'
MODEL_PATH = 'food_classifier_model.h5'
DATA_PROCESSED_PATH = 'processed_data.pkl'

# Tạo thư mục lưu trữ
os.makedirs('dataset', exist_ok=True)
for food_class in FOOD_CLASSES:
    os.makedirs(f'dataset/{food_class}', exist_ok=True)

print(f" Sẽ phân loại {len(FOOD_CLASSES)} loại món ăn:")
for i, food in enumerate(FOOD_CLASSES):
    print(f"   {i+1}. {food}")

def check_existing_model():
    """
    Kiểm tra xem đã có mô hình và dữ liệu được huấn luyện chưa
    """
    model_exists = os.path.exists(MODEL_PATH)
    data_exists = os.path.exists(DATA_PROCESSED_PATH)
    
    print(" Kiểm tra mô hình đã học:")
    print(f"    Mô hình AI: {' Có' if model_exists else ' Chưa có'}")
    print(f"    Dữ liệu xử lý: {' Có' if data_exists else ' Chưa có'}")
    
    if model_exists and data_exists:
        print(" Đã tìm thấy mô hình đã học! Sẽ load mô hình cũ.")
        return True
    else:
        print(" Chưa có mô hình, sẽ bắt đầu học từ đầu.")
        return False

def generate_synthetic_food_image_bw(food_type, image_index, size=(60, 60)):
    """
    Tạo ảnh ĐEN TRẮNG cho từng loại món ăn với đặc trưng riêng
    """
    # Tạo ảnh grayscale với giá trị đặc trưng cho từng món
    gray_patterns = {
        'Phở': {
            'base_gray': 120,    # Xám trung bình (nước phở)
            'accent_grays': [200, 80, 160, 40]  # Các mức xám khác nhau
        },
        'Bún Bò': {
            'base_gray': 80,     # Xám đậm (nước bún bò)
            'accent_grays': [180, 60, 140, 30]
        },
        'Bánh Xèo': {
            'base_gray': 200,    # Xám sáng (bánh xèo)
            'accent_grays': [100, 150, 70, 220]
        },
        'Cơm Tấm': {
            'base_gray': 240,    # Xám rất sáng (cơm trắng)
            'accent_grays': [120, 180, 90, 60]
        },
        'Bánh Mì': {
            'base_gray': 160,    # Xám vừa (bánh mì)
            'accent_grays': [100, 200, 140, 80]
        }
    }
    
    # Tạo ảnh base grayscale
    img = np.ones((size[0], size[1]), dtype=np.uint8)
    pattern = gray_patterns[food_type]
    base_gray = pattern['base_gray']
    
    # Thêm nhiễu và biến thể cho base gray
    for i in range(size[0]):
        for j in range(size[1]):
            noise = np.random.randint(-20, 20)
            gray_value = np.clip(base_gray + noise, 0, 255)
            img[i, j] = gray_value
    
    # Thêm các chi tiết đặc trưng bằng các mức xám khác
    accent_grays = pattern['accent_grays']
    num_details = random.randint(8, 20)
    
    for _ in range(num_details):
        # Random position và size
        x = random.randint(2, size[0]-8)
        y = random.randint(2, size[1]-8)
        w = random.randint(2, 6)
        h = random.randint(2, 6)
        
        # Random gray value từ accent grays
        gray_value = random.choice(accent_grays)
        img[x:x+w, y:y+h] = gray_value
    
    # Thêm texture đặc trưng
    if random.random() > 0.4:
        # Thêm đường kẻ (như sợi phở, bún)
        if food_type in ['Phở', 'Bún Bò']:
            for _ in range(random.randint(5, 12)):
                start_y = random.randint(0, size[1])
                end_y = random.randint(0, size[1])
                x_pos = random.randint(0, size[0]-1)
                gray_line = random.choice(accent_grays)
                cv2.line(img, (start_y, x_pos), (end_y, x_pos), gray_line, thickness=1)
        
        # Thêm hình tròn/oval cho một số món
        elif food_type in ['Bánh Xèo', 'Cơm Tấm']:
            for _ in range(random.randint(3, 8)):
                center_x = random.randint(10, size[0]-10)
                center_y = random.randint(10, size[1]-10)
                radius = random.randint(2, 8)
                gray_circle = random.choice(accent_grays)
                cv2.circle(img, (center_y, center_x), radius, gray_circle, -1)
    
    return img

def collect_training_data():
    """
    Thu thập dữ liệu huấn luyện bằng cách tạo ảnh ĐEN TRẮNG
    """
    print(" Bắt đầu thu thập dữ liệu huấn luyện...")
    print(f" Sẽ tạo {NUM_IMAGES_PER_CLASS} ảnh ĐEN TRẮNG cho mỗi lớp")
    
    total_images = 0
    
    for class_idx, food_class in enumerate(FOOD_CLASSES):
        print(f"\n Đang tạo ảnh ĐEN TRẮNG cho {food_class}...")
        
        for img_idx in range(NUM_IMAGES_PER_CLASS):
            # Tạo ảnh ĐEN TRẮNG
            synthetic_img_bw = generate_synthetic_food_image_bw(food_class, img_idx)
            
            # Lưu ảnh (ảnh đã là grayscale)
            img_path = f'dataset/{food_class}/img_{img_idx:03d}.jpg'
            cv2.imwrite(img_path, synthetic_img_bw)
            
            total_images += 1
            
            # Hiển thị tiến trình
            if (img_idx + 1) % 10 == 0:
                print(f"    Đã tạo {img_idx + 1}/{NUM_IMAGES_PER_CLASS} ảnh ĐEN TRẮNG")
    
    print(f"\n Hoàn thành! Đã tạo tổng cộng {total_images} ảnh ĐEN TRẮNG")
    
    # Hiển thị một số ảnh mẫu ĐEN TRẮNG
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(' Mẫu ảnh ĐEN TRẮNG đã tạo (60x60)', fontsize=16)
    
    for i, food_class in enumerate(FOOD_CLASSES):
        img_path = f'dataset/{food_class}/img_000.jpg'
        img_bw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        axes[i].imshow(img_bw, cmap='gray')
        axes[i].set_title(f'{food_class}', fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return total_images


def load_and_preprocess_data():
    """
    Load và xử lý dữ liệu: đã là grayscale 60x60, chỉ cần normalize
    """
    print(" Đang load và xử lý dữ liệu...")
    
    X = []  # Dữ liệu ảnh
    y = []  # Nhãn
    
    for class_idx, food_class in enumerate(FOOD_CLASSES):
        class_path = f'dataset/{food_class}'
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f" {food_class}: {len(image_files)} ảnh ĐEN TRẮNG")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            
            # Đọc ảnh grayscale (đã là 60x60)
            img_bw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Normalize về [0, 1]
            img_normalized = img_bw.astype(np.float32) / 255.0
            
            X.append(img_normalized)
            y.append(class_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f" Đã xử lý xong!")
    print(f" Shape dữ liệu: X = {X.shape}, y = {y.shape}")
    
    # Lưu dữ liệu đã xử lý
    with open(DATA_PROCESSED_PATH, 'wb') as f:
        pickle.dump((X, y), f)
    print(f" Đã lưu dữ liệu xử lý vào {DATA_PROCESSED_PATH}")
    
    return X, y

def load_processed_data():
    """
    Load dữ liệu đã xử lý từ file
    """
    print(f" Đang load dữ liệu từ {DATA_PROCESSED_PATH}...")
    with open(DATA_PROCESSED_PATH, 'rb') as f:
        X, y = pickle.load(f)
    print(f" Đã load xong! Shape: X = {X.shape}, y = {y.shape}")
    return X, y


def create_cnn_model():
    """
    Tạo mô hình CNN để phân loại ảnh ĐEN TRẮNG
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(60, 60, 1)),
        
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Flatten và Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(FOOD_CLASSES), activation='softmax')
    ])
    
    return model

# Kiểm tra mô hình đã tồn tại
has_existing_model = check_existing_model()

if has_existing_model:
    # LOAD MÔ HÌNH CŨ
    print("\n ĐANG LOAD MÔ HÌNH ĐÃ HỌC...")
    print("="*50)
    
    # Load mô hình
    model = keras.models.load_model(MODEL_PATH)
    print(f" Đã load mô hình từ {MODEL_PATH}")
    
    # Load dữ liệu đã xử lý
    X, y = load_processed_data()
    
    # Chia dữ liệu để đánh giá
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Reshape cho CNN
    X_test_cnn = X_test.reshape(-1, 60, 60, 1)
    
    # Đánh giá nhanh
    test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f" Độ chính xác mô hình đã học: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(" MÔ HÌNH SẴN SÀNG SỬ DỤNG!")
    
else:
    # TRAIN MÔ HÌNH MỚI
    print("\n BẮT ĐẦU HỌC MỚI...")
    print("="*50)
    
    # Thu thập dữ liệu
    total_collected = collect_training_data()
    
    # Load và xử lý dữ liệu
    X, y = load_and_preprocess_data()
    
    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f" Train set: {X_train.shape[0]} ảnh")
    print(f" Test set: {X_test.shape[0]} ảnh")
    
    # Tạo mô hình
    print(" Đang xây dựng mô hình CNN...")
    model = create_cnn_model()
    
    # Compile mô hình
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Hiển thị cấu trúc mô hình
    model.summary()
    
    # HUẤN LUYỆN MÔ HÌNH
    # Reshape dữ liệu cho CNN
    X_train_cnn = X_train.reshape(-1, 60, 60, 1)
    X_test_cnn = X_test.reshape(-1, 60, 60, 1)
    
    print(" Bắt đầu huấn luyện mô hình...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Callback để theo dõi quá trình huấn luyện
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.001
        )
    ]
    
    # Huấn luyện
    history = model.fit(
        X_train_cnn, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test_cnn, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    print(" Hoàn thành huấn luyện!")
    
    # LƯU MÔ HÌNH
    model.save(MODEL_PATH)
    print(f" Đã lưu mô hình vào {MODEL_PATH}")
    
    # Đánh giá mô hình
    test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f" Độ chính xác trên test set: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Vẽ biểu đồ quá trình huấn luyện
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(' Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(' Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def preprocess_test_image_bw(image_path):
    """
    Xử lý ảnh test: chuyển ĐEN TRẮNG, resize 60x60, normalize
    """
    # Đọc ảnh
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError("Không thể đọc ảnh!")
    
    # Chuyển sang grayscale NGAY từ đầu
    if len(img.shape) == 3:  # Nếu là ảnh màu
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # Nếu đã là grayscale
        img_gray = img
    
    # Resize về 60x60
    img_resized = cv2.resize(img_gray, IMAGE_SIZE)
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img, img_gray, img_resized, img_normalized

def predict_food_image_bw(image_path, show_result=True):
    """
    Dự đoán loại món ăn từ ảnh ĐEN TRẮNG
    """
    try:
        # Xử lý ảnh
        img_original, img_gray, img_resized, img_processed = preprocess_test_image_bw(image_path)
        
        # Reshape cho mô hình
        img_input = img_processed.reshape(1, 60, 60, 1)
        
        # Dự đoán
        predictions = model.predict(img_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100
        
        predicted_food = FOOD_CLASSES[predicted_class_idx]
        
        if show_result:
            # Hiển thị kết quả
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Ảnh gốc
            if len(img_original.shape) == 3:
                img_display = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
                axes[0].imshow(img_display)
            else:
                axes[0].imshow(img_original, cmap='gray')
            axes[0].set_title(' Ảnh gốc', fontsize=14)
            axes[0].axis('off')
            
            # Ảnh grayscale gốc
            axes[1].imshow(img_gray, cmap='gray')
            axes[1].set_title(' Ảnh ĐEN TRẮNG', fontsize=14)
            axes[1].axis('off')
            
            # Ảnh cuối cùng (60x60)
            axes[2].imshow(img_resized, cmap='gray')
            axes[2].set_title(' 60x60 ĐEN TRẮNG', fontsize=14)
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Kết quả dự đoán
            print("="*50)
            print(" KẾT QUẢ DỰ ĐOÁN")
            print("="*50)
            print(f" Món ăn được dự đoán: {predicted_food}")
            print(f" Độ tin cậy: {confidence:.2f}%")
            print("="*50)
            
            # Hiển thị tất cả xác suất
            print("\n Chi tiết độ tin cậy cho từng món:")
            for i, food in enumerate(FOOD_CLASSES):
                prob = predictions[0][i] * 100
                bar = "█" * int(prob // 2)
                print(f"{food:12}: {prob:6.2f}% {bar}")
        
        return predicted_food, confidence, predictions[0]
    
    except Exception as e:
        print(f" Lỗi: {str(e)}")
        return None, 0, None

def test_image_from_computer():
    """
    Chọn ảnh từ máy tính và phân loại món ăn
    """
    print("\n CHỌN ẢNH TỪ MÁY TÍNH")
    print("="*50)
    print(" Hướng dẫn:")
    print("    Sẽ mở hộp thoại chọn file")
    print("    Chọn ảnh món ăn từ máy tính")
    print("    Ảnh sẽ được tự động chuyển thành ĐEN TRẮNG và resize 60x60")
    print("    AI sẽ dự đoán loại món ăn")
    print("="*50)
    
    # Mở hộp thoại chọn file
    selected_file = select_image_file()
    
    if selected_file:
        # Hiển thị thông tin file
        show_file_info(selected_file)
        
        # Phân loại ảnh
        print(" Đang phân tích ảnh...")
        result = predict_food_image_bw(selected_file)
        
        if result[0]:
            print(f" Phân tích thành công!")
        else:
            print(" Có lỗi xảy ra trong quá trình phân tích!")
            
    else:
        print(" Không có file nào được chọn!")

def test_multiple_images():
    """
    Test nhiều ảnh liên tiếp
    """
    print("\n🖼 TEST NHIỀU ẢNH LIÊN TIẾP")
    print("="*50)
    
    while True:
        print("\n Chọn ảnh tiếp theo để test:")
        test_image_from_computer()
        
        # Hỏi có muốn tiếp tục không
        print("\n" + "="*30)
        choice = input(" Bạn có muốn test thêm ảnh khác không? (y/n): ").lower().strip()
        
        if choice != 'y' and choice != 'yes':
            break
    
    print(" Cảm ơn bạn đã sử dụng phần mềm!")


print("\n" + "="*60)
print(" PHẦN MỀM PHÂN LOẠI MÓN ĂN VIỆT NAM SẴN SÀNG!")
print("="*60)
print(" Tóm tắt:")
if not has_existing_model:
    print(f"   • Đã tạo {NUM_IMAGES_PER_CLASS * len(FOOD_CLASSES)} ảnh ĐEN TRẮNG")
print(f"   • Tất cả ảnh đều được xử lý ĐEN TRẮNG 60x60")
print(f"   • Mô hình đã được lưu vào: {MODEL_PATH}")
print(f"   • Có thể phân loại {len(FOOD_CLASSES)} món ăn:")
for i, food in enumerate(FOOD_CLASSES):
    print(f"     {i+1}. {food}")
print("="*60)

print("\n CÁCH SỬ DỤNG:")
print("    Test 1 ảnh: test_image_from_computer()")
print("    Test nhiều ảnh: test_multiple_images()")
print("\n Lần chạy tiếp theo sẽ tự động load mô hình đã học!")

print("\n" + "="*60)
print(" BẮT ĐẦU TEST NGAY BÂY GIỜ!")
print("="*60)

# Tự động bắt đầu test ảnh
test_image_from_computer()
