#Bước 1: Thu thập dữ liệu khuôn mặt từ camera
# 
#-	Code:
import cv2
import os
import time

# Khởi tạo camera
cam = cv2.VideoCapture(0)

# Sử dụng bộ phát hiện khuôn mặt từ OpenCV
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Nhập tên người dùng
while True:
    face_name = input('\nNhập tên người dùng: ').strip()
    file_path = 'dataset/labels.txt'

    # Tạo thư mục 'dataset' nếu chưa tồn tại
    if not os.path.exists('dataset'):
        os.mkdir('dataset')

    # Tạo thư mục theo tên người dùng
    user_folder = f'dataset/{face_name}'
    if not os.path.exists(user_folder):
        os.mkdir(user_folder)
        break
    else:
        print('\nTên người dùng này đã tồn tại, vui lòng nhập tên khác!')

print("\n[THÔNG BÁO] Đang khởi tạo tính năng chụp khuôn mặt. Nhìn vào camera và chờ đợi...")

# Khởi tạo biến lưu trữ số lượng ảnh khuôn mặt
count = 0

# Kiểm tra xem tệp nhãn đã tồn tại hay chưa
file_exists = os.path.exists(file_path)

# Ghi tên người dùng vào file labels.txt
with open(file_path, 'a') as file:
    if file_exists:
        file.write('\n')
    file.write(face_name)

# Các biến tính toán FPS
frame_count = 0
start_time = time.time()
fps = 0  # Khởi tạo biến FPS

# Bắt đầu vòng lặp chụp ảnh khuôn mặt
while True:
    # Đọc hình ảnh từ camera
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Lật hình ảnh theo chiều ngang
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong hình ảnh
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Cắt và lưu ảnh chỉ chứa khuôn mặt
        cropped_face = gray[y:y + h, x:x + w]
        try:
            cv2.imwrite(f"{user_folder}/face_{face_name}_{count}.jpg", cropped_face)
            count += 1
        except Exception as e:
            print(f"Lỗi khi lưu ảnh: {e}")
            continue

    # Tính toán FPS
    frame_count += 1
    if frame_count >= 10:  # Cập nhật FPS mỗi 10 khung hình
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Hiển thị FPS
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị số lượng ảnh khuôn mặt đã chụp
    text_position = (img.shape[1] - 210, 30)
    cv2.putText(img, f"Số ảnh: {count}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị hình ảnh từ camera
    cv2.imshow('Camera', img)

    # Dừng khi nhấn 'ESC' hoặc đã chụp đủ 500 ảnh
    k = cv2.waitKey(20) & 0xff
    if k == 27 or count >= 500:
        break

print("\n[THÔNG BÁO] Đã hoàn thành quá trình thu thập dữ liệu!")

# Giải phóng tài nguyên
cam.release()
cv2.destroyAllWindows()


#Bước 2: Thực hiện quá trình huấn luyện mô hình nhận diện khuôn mặt sử dụng thư viện Keras
#Code:
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
from model import build  # Import hàm xây dựng mô hình từ file 'model.py'

# Các thông số ban đầu
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96, 96, 3)  # Kích thước ảnh: 96x96 pixels, 3 kênh màu (RGB)

data = []
labels = []

# Nạp tệp ảnh từ tập dữ liệu
image_files = [
    f for f in glob.glob('dataset/**/*', recursive=True)
    if not os.path.isdir(f) and not f.endswith('labels.txt')
]
random.shuffle(image_files)

# Mở tệp 'labels.txt' để đọc nhãn
with open('dataset/labels.txt', 'r') as file:
    _labels = file.read().splitlines()  # Đọc từng dòng làm nhãn

# Chuyển đổi hình ảnh thành mảng và gán nhãn
for img_path in image_files:
    image = cv2.imread(img_path)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    # Lấy nhãn từ tên thư mục cha
    label = img_path.split(os.path.sep)[-2]
    labels.append(_labels.index(label))

# Tiền xử lý dữ liệu
data = np.array(data, dtype="float") / 255.0  # Chuẩn hóa dữ liệu ảnh
labels = np.array(labels)

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Chuyển đổi nhãn thành dạng one-hot encoding
trainY = to_categorical(trainY, num_classes=len(_labels))
testY = to_categorical(testY, num_classes=len(_labels))

# Tạo thêm dữ liệu mới từ tập dữ liệu hiện có
aug = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Xây dựng mô hình
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=len(_labels))

# Biên dịch mô hình
opt = Adam(learning_rate=lr, decay=lr / epochs)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Huấn luyện mô hình
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batch_size,
    epochs=epochs,
    verbose=1
)

# Lưu mô hình đã huấn luyện
model.save('D:/face_detection.keras')
print("[THÔNG BÁO] Mô hình đã được lưu tại D:/face_detection.keras")

# Vẽ biểu đồ loss và accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs

# Loss
plt.plot(np.arange(0, N), H.history["loss"], label="Train Loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="Validation Loss")

# Accuracy
plt.plot(np.arange(0, N), H.history["accuracy"], label="Train Accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="Validation Accuracy")

# Thông tin biểu đồ
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# Lưu biểu đồ
plt.savefig('plot.png')
print("[THÔNG BÁO] Biểu đồ loss và accuracy đã được lưu tại plot.png")

#Bước 3: Ứng dụng thực tế sử dụng mô hình nhận diện khuôn mặt đã được huấn luyện để phát hiện và nhận dạng khuôn mặt từ webcam
#-	Code:
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2

# Sử dụng CascadeClassifier để tải bộ phân loại khuôn mặt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Tải mô hình nhận diện khuôn mặt đã được huấn luyện
model = load_model('D:/face_detection.keras')

# Mở webcam
webcam = cv2.VideoCapture(0)

# Đọc nhãn từ tệp 'labels.txt'
with open('dataset/labels.txt', 'r') as file:
    classes = file.read().splitlines()

# Lặp qua từng khung hình từ webcam
while webcam.isOpened():
    # Đọc khung hình từ webcam
    status, frame = webcam.read()
    if not status:
        print("[THÔNG BÁO] Không thể đọc khung hình từ webcam.")
        break

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )

    # Xử lý từng khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cắt vùng khuôn mặt
        face_crop = frame[y:y + h, x:x + w]

        # Kiểm tra kích thước vùng khuôn mặt
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Tiền xử lý vùng khuôn mặt
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Dự đoán nhãn và độ chính xác
        conf = model.predict(face_crop)[0]  # Trả về ma trận 2D
        index = np.argmax(conf)
        label = classes[index]

        # Xác định "Unknown" nếu độ chính xác < 0.7
        if conf[index] < 0.7:
            label = "Unknown"
        else:
            label = "{}: {:.2f}%".format(label, conf[index] * 100)

        # Hiển thị nhãn trên khung hình
        Y = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(frame, label, (x, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị khung hình với khuôn mặt được nhận diện
    cv2.imshow("Face Detection", frame)

    # Thoát khi nhấn phím "ESC"
    if cv2.waitKey(20) & 0xFF == 27:
        break

# Giải phóng tài nguyên
webcam.release()
cv2.destroyAllWindows()
print("[THÔNG BÁO] Đã giải phóng webcam và đóng ứng dụng.")

#3.3.2	Test script
#  		Sử dụng các framework kiểm thử như unittest (cho Python) hoặc các thư viện khác như pytest:
#Code:
import unittest
import cv2
import os
from your_program_file import your_functions_or_classes  # Import các hàm hoặc lớp từ chương trình của bạn

class TestYourProgram(unittest.TestCase):
    def setUp(self):
        """
        Thiết lập môi trường kiểm thử trước mỗi test case.
        """
        if not os.path.exists('dataset'):
            os.mkdir('dataset')
        # Tạo tệp nhãn giả để kiểm tra
        with open('dataset/labels.txt', 'w') as file:
            file.write('John\n')

    def tearDown(self):
        """
        Dọn dẹp môi trường kiểm thử sau mỗi test case.
        """
        if os.path.exists('dataset/labels.txt'):
            os.remove('dataset/labels.txt')
        if os.path.exists('dataset'):
            os.rmdir('dataset')

    def test_invalid_input_name(self):
        """
        Kiểm tra xử lý khi tên người dùng bị để trống.
        """
        with self.assertRaises(ValueError):
            your_functions_or_classes.validate_user_input("")

    def test_duplicate_user_name(self):
        """
        Kiểm tra xử lý khi tên người dùng đã tồn tại.
        """
        with self.assertRaises(ValueError):
            your_functions_or_classes.validate_user_input("John")

    def test_image_capture_process(self):
        """
        Kiểm tra quá trình chụp ảnh khuôn mặt.
        """
        result = your_functions_or_classes.capture_face_images()
        self.assertEqual(result, "Đã hoàn thành quá trình thu thập dữ liệu!")

    def test_read_labels_file(self):
        """
        Kiểm tra chức năng đọc dữ liệu từ tệp nhãn.
        """
        labels = your_functions_or_classes.read_labels_file("dataset/labels.txt")
        self.assertIsInstance(labels, list)
        self.assertTrue(all(isinstance(label, str) for label in labels))
        self.assertEqual(labels, ["John"])  # Kiểm tra nội dung tệp ban đầu

if __name__ == '__main__':
    unittest.main()

