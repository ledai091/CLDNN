import numpy as np
import random
from data_aug_tech import DataAug
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import utils
from copy import deepcopy

class DataGenerator(utils.Sequence):
    # Lớp này dùng để đọc và đưa dữ liệu vào model. Kế thừa từ lớp utils.Sequence trong Keras
    def __init__(self, list_IDs, path2data, encode_label, batch_size, img_size, n_channels, n_classes, shuffle, is_aug):
        # Khởi tạo:
        #   list_IDs: list các đường dẫn vào ảnh
        #   path2data: Đường dẫn vào thư mục của các lớp
        #   encode_label: Từ diển mã hóa lớp thành số. Dùng để tạo one-hot vector
        #   img_size: Kích thước ảnh
        #   n_channels: Số kênh của ảnh
        #   n_classes: Số lớp trong dữ liệu
        #   shuffle: Chọn có trộn dữ liệu khi huấn luyện hay không.
        #   is_aug: Chọn có tăng cường ảnh hay không

        self.list_IDs = deepcopy(list_IDs[:])
        self.img_size = img_size
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.pathy = path2data
        self.c1h = encode_label
        self.is_aug = is_aug
        self.DataAug = DataAug()

        self.on_epoch_end()

    def __len__(self):
        # Định ra tổng số batch trong 1 epoch. Cần lưu ý rằng model chỉ gọi hàm này đúng một lần để xác định tổng số bước
        # trong 1 epoch. Do đó, nếu bạn có làm những phương pháp đòi hỏi biến đổi số bước qua từng batch (ví dụ như adapt
        # batch hay YMufT) thì bạn hãy cẩn thận, vì có thể model sẽ báo rằng model không chạy đủ bước, bởi vì model chỉ
        # lưu thông số ở epoch đầu tiên.
        # Nếu bạn muốn sửa điều này thì bạn phải sửa trên file gốc của Keras. Cân nhắc khi thực hiện điều này. Xem tại
        # https://github.com/keras-team/keras/issues/10615#issuecomment-664031073
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Tạo dữ liệu cho 1 batch'
        # Lấy chỉ số của hình để đọc hình
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Từ cách chỉ số đã chọn bên trên, ta lấy đường dẫn ảnh tương ứng
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # X là batch, chứa batch_size ảnh. y là one-hot vector tương ứng
        X, y = self.__data_generation(list_IDs_temp)

        # Chuẩn hóa [0, 1]
        X /= 255.0

        return X, y

    def on_epoch_end(self):
        'Hàm này được model gọi khi kết thúc 1 epoch'
        # Đánh chỉ số các đường dẫn ảnh
        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:
            # Trộn ảnh bằng cách trộn đường dẫn
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Gọi ảnh, số lượng bằng với số batch'

        # X là ma trận chứa các ma trận ảnh, có kích thước là kích thước batch x dài ảnh x rộng ảnh x số kênh ảnh
        X = np.empty((self.batch_size, self.img_size[0], self.img_size[1], self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=float)

        # Tải ảnh
        for i, ID in enumerate(list_IDs_temp):
            # Gán đường dẫn đầy đủ vào biến pata
            pata = self.pathy + '/' + ID['class'] + '/' + ID['name_img']

            img = load_img(pata, target_size=(self.img_size[0], self.img_size[1]))
            if self.is_aug:
                # Nếu ta chọn tăng cường dữ liệu thì model sẽ đọc những kỹ thuật tăng cường dữ liệu ta đã chọn.
                # Trong ví dụ này, ta có 2 cách tăng cường: Xoay ảnh và zoom ảnh. Model chọn ngẫu nhiên 1 trong 2.
                img = img_to_array(img)
                if random.randint(0, 9) % 2 == 0:
                    img = self.DataAug.random_rotation(x=img, rg=360, channel_axis=self.n_channels)
                else:
                    img = self.DataAug.random_zoom(x=img, zoom_range=[0.5, 2.0], channel_axis=self.n_channels)

            X[i,] = img_to_array(img)
            del img

            # Tạo one-hot vector
            onehot_class = np.zeros(self.n_classes, dtype=float)
            onehot_class[self.c1h[ID['class']]] = 1.0
            y[i] = onehot_class

        return X, y