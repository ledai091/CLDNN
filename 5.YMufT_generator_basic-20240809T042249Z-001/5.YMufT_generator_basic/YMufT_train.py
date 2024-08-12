import numpy as np
import gc

from copy import deepcopy
from random import shuffle
from data_aug_tech import DataAug
from data_generator_classification import DataGenerator

from tensorflow.keras.layers import Dense
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD

class checkingPerformance():
    # Đây là lớp chính, dùng để train và test modle theo kiểu YMufT. Hàm stat_species, arrange_data và YMufT có ý nghĩa
    # như bên file train_model.py
    def __init__(self, path2data,class_encode, lr, img_width, img_height, img_channels, num_classes):
        self.path2data = path2data
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.class_encode = class_encode
        self.lr = lr

        self.DataAug = DataAug()

    def get_model(self):
        # Xài transfer learning từ imagenet. Xài Mobilenet phiên bản 2
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=None, pooling='avg',
                                 input_shape=(self.img_width, self.img_height, 3), classes=self.num_classes)
        x = base_model.output
        x = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=x)
        print(model.summary())
        print('MobileNetV2')
        return model

    def stat_species(self, type_ori, class_encode):
        tke = []
        for _ in class_encode:
            tke.append([])

        lst_class = list(class_encode)
        for elem in type_ori:
            ind = lst_class.index(elem['class'])
            tke[ind].append(elem)

        return tke

    def arrange_data(self):
        num_data_each_class = np.zeros(self.num_classes)
        for num_class in range(self.num_classes):
            num_data_each_class[num_class] = len(self.list_IDs[num_class])

        return num_data_each_class

    def YMufT(self):
        A = deepcopy(self.list_IDs[:])
        B = deepcopy(self.lst_ratio[:])

        if ~np.any(self.B_temp):
            # Trường hợp đã chia hết toàn bộ dữ liệu vào các fold thì khôi phục dữ liệu gốc, chia lại từ đầu
            print('end')
            self.A_temp = deepcopy(self.list_IDs[:])
            self.B_temp = deepcopy(self.lst_ratio[:])
            gc.collect()

        MC = np.where(self.B_temp > 0, self.B_temp, np.inf).argmin()
        eps = 0.5 * self.B_temp[MC]
        bou1 = np.where(self.B_temp <= self.B_temp[MC]+eps)[0]
        bou2 = self.B_temp[bou1]
        MB = self.B_temp[bou1[np.argmax(bou2)]]

        F = []
        for i in range (0, self.num_classes):
            if self.B_temp[i] > 0:
                nt = int(np.minimum(self.B_temp[i], MB))
                shuffle(self.A_temp[i])
                F += self.A_temp[i][:nt]
                del self.A_temp[i][:nt]
                self.B_temp[i] -= nt
            else:
                shuffle(A[i])
                nt = int(np.minimum(B[i], MB))
                F += A[i][:nt]

        return F

    def training(self, train_ori, batch_train, val_ori, batch_val, class_encode, num_loop_eps, total_fold, is_aug):
        # Huấn luyện model.
        clear_session()

        self.list_IDs = self.stat_species(train_ori, class_encode)
        self.lst_ratio = self.arrange_data()
        self.A_temp = deepcopy(self.list_IDs[:])
        self.B_temp = deepcopy(self.lst_ratio[:])

        #Tạo generator cho tập validate. Tậo val thì không cần chia YMufT
        val_gen = DataGenerator(list_IDs=val_ori, path2data=self.path2data, encode_label=self.class_encode,
                                  batch_size=batch_val, img_size=(self.img_width, self.img_height),
                                  n_channels=self.img_channels, n_classes=self.num_classes, shuffle=False, is_aug=False)

        self.model = self.get_model()
        self.model.compile(optimizer=SGD(lr=self.lr, momentum=0.9),
                           loss='categorical_crossentropy', metrics=['accuracy'])

        loss_global = np.inf
        acc_global = 0
        # File save_history.txt dùng để lưu thông số, như số epoch, acc val, loss vall, acc train, loss train để vẽ biểu
        # đồ sau khi train
        f = open('save_history.txt', 'w')
        count_eps = 0

        # Bắt đầu huấn luyện. Thực tế, khi code thì lượt chu kỳ sẽ đếm ngược từ chu kỳ cuối num_loop_eps về chu kỳ đầu 1
        # bởi vì số vòng lặp mỗi fold đi từ num_loop_eps xuống 1 qua mỗi chu kỳ.
        for training_periods in range(num_loop_eps, 0, -1):
            for fold in range(total_fold):
                print('Training period: %d, fold: %d' %(num_loop_eps - training_periods + 1, fold + 1))

                # F chính là dữ liệu fold thứ fold
                F = self.YMufT()

                # Sẽ có trường hợp kích thước fold nhỏ hơn kích thước batch train. Khi đó ta thấy kích thước batch bằng
                # với kích thước fold
                batch_size = np.minimum(batch_train, len(F))

                train_gen = DataGenerator(list_IDs = F, path2data=self.path2data, encode_label=self.class_encode,
                                          batch_size= batch_size, img_size = (self.img_width, self.img_height),
                                          n_channels=self.img_channels, n_classes=self.num_classes, shuffle=True,
                                          is_aug=is_aug)
                for _ in range (training_periods):
                    count_eps +=1
                    hist = self.model.fit(train_gen, validation_data=val_gen, verbose=1, epochs=1, batch_size=batch_train)
                    acc_val = hist.history['val_accuracy'][0]
                    loss_val = hist.history['val_loss'][0]

                    f.write("epoch %d train_acc %.5f train_loss %.5f val_acc %.5f val_loss %.5f\n"
                            %(count_eps, hist.history['accuracy'][0], hist.history['loss'][0], acc_val, loss_val))
                    print('Validation loss = %.5f, Validation accuracy = %.5f' % (loss_val, acc_val))

                    # Lưu model có độ chính xác trên tập val cao nhất. Nếu độ chính xác bằng nhau thì lưu modle có độ
                    # lỗi nhỏ hơn
                    if acc_global < acc_val:
                        print('Validation accuracy improved from %.5f to %.5f' % (acc_global, acc_val))
                        acc_global = acc_val
                        loss_global = loss_val
                        self.model.save('mobilenetv2')
                    elif acc_global == acc_val and loss_global > loss_val:
                        print('Validation loss reduced from %.5f to %.5f' % (loss_global, loss_val))
                        acc_global = acc_val
                        loss_global = loss_val
                        self.model.save('mobilenetv2')
                    gc.collect()
                del train_gen, F, hist
                gc.collect()

                print('BEST-SO-FAR: Validation loss = %.5f, Validation accuracy = %.5f' % (loss_global, acc_global))
        f.close()
        self.plot_perform()

    def testing(self, test_ori, batch_test):
        # Kiểm kết qua 3huấn luyện bằng tập test
        model = load_model('mobilenetv2')
        test_gen = DataGenerator(list_IDs=test_ori, path2data=self.path2data, encode_label=self.class_encode,
                                  batch_size=batch_test, img_size=(self.img_width, self.img_height),
                                  n_channels=self.img_channels, n_classes=self.num_classes, shuffle=False, is_aug=False)
        result = model.evaluate(test_gen)
        print('Loss: ', result[0])
        print('Accuracy: ', result[1])

    def plot_perform(self):
        # In biểu đồ huấn luyện
        import matplotlib.pyplot as plt
        with open('save_history.txt', 'r') as f:
            content = f.readlines()
        f.close()

        train_acc = []
        train_loss = []
        val_acc = []
        val_loss = []
        for x in content:
            kiki = x.strip().split()
            n = len(kiki)
            for i in range(0, n, 2):
                if kiki[i] == 'train_acc':
                    train_acc.append(float(kiki[i + 1]))
                elif kiki[i] == 'train_loss':
                    train_loss.append(float(kiki[i + 1]))
                elif kiki[i] == 'val_acc':
                    val_acc.append(float(kiki[i + 1]))
                elif kiki[i] == 'val_loss':
                    val_loss.append(float(kiki[i + 1]))

        epochs = range(len(train_acc))
        plt.plot(epochs, train_acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.xticks(epochs)
        plt.title('Training and validation accuracy')
        plt.tight_layout()
        plt.legend()
        fig1 = plt.gcf()
        fig1.savefig("acc.png", dpi=300)
        plt.clf()

        plt.plot(epochs, train_loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.xticks(epochs)
        plt.title('Training and validation loss')
        plt.tight_layout()
        plt.legend()
        fig1 = plt.gcf()
        fig1.savefig("loss.png", dpi=300)
        plt.clf()