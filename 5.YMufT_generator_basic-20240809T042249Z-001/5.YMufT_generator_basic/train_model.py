import pickle
import gc
import numpy as np
from random import shuffle
from copy import deepcopy
from YMufT_train import checkingPerformance

# File huấn luyện model theo YMufT. Trong file này, bạn coi hàm performModel để hiểu quy trình huấn luyện

def stat_species(type_ori, class_encode):
    tke = []
    for i in class_encode:
        tke.append([])

    lst_class = list(class_encode)
    for elem in type_ori:
        ind = lst_class.index(elem['class'])
        tke[ind].append(elem)

    return tke

def arrange_data(list_IDs, n_classes):
    num_data_each_class = np.zeros(n_classes)
    for num_class in range(n_classes):
        num_data_each_class[num_class] = len(list_IDs[num_class])

    return num_data_each_class

def collect_data_ymuft(type_ori, class_encode):
    # Người dùng vui lòng đọc Thuật toán 1 (Algorithm 1) trong bài báo https://www.mdpi.com/2076-3417/11/8/3331
    # Hàm này chia fold theo YMufT. Hàm lấy input là tập các đường dẫn (dạng tư điển) của tập huấn luyện (type_dict) và
    # từ điển mã hóa lớp (class_encode)
    # list_IDs chia các đường dẫn vô các lớp tương ứng. Tức là list_IDs là một list chừa các list con là các lớp, mỗi
    # list con i chứa tất cả tập các đường dẫn (dạng từ điển) của các ảnh trong lớp i.
    # lst_ratio đếm số lưởng ảnh ở mỗi lớp. Biến này dùng để đếm ảnh khi chia dữ liệu.
    sumsum = 0
    list_IDs = stat_species(type_ori, class_encode)
    n_classes = len(class_encode)
    lst_ratio = arrange_data(list_IDs, n_classes)

    # A, B, A_temp và B_temp để xác định đâu là dữ liệu gốc, đâu là dữ liệu chia. trong YMufT, nếu 1 lớp đã chia hết dữ
    # liệu thì ta khôi phục dữ liệu gốc để chia
    A = deepcopy(list_IDs[:])
    B = deepcopy(lst_ratio[:])
    A_temp = deepcopy(A)
    B_temp = deepcopy(B)

    # Biến total_fold là list chứa các list con chính là các fold, mỗi fold là 1 list chứa các đường dẫn tương ứng
    total_fold = []
    while np.any(B_temp):
        # Lặp cho tới khi tất cả dữ liệu đều được chia vào các fold. Tức không còn giá trị trong B_temp dương. Mỗi vòng
        # lặp là 1 fold mới
        # Số 0.5 khi gán biến eps có nghĩa tỉ lệ xác định số lượng dữ liệu 2 lớp xấp xỉ nhau. Ví dụ lớp hoa Mai có 10
        # tấm, khi đó những lớp có tối đa 0.5*10 + 10 = 15 được coi là xấp xỉ với lớp hoa Mai, và các lớp này
        # (và lớp hoa Mai) sẽ nằm chung 1 fold. Bạn có thể điều chỉnh con số này tùy ý.
        # Biến F là biến chứa tất cả dữ liệu trong 1 fold.
        # MC xác định trong những lớp vẫn còn dữ liệu chưa được chia vào fold, lớp nào có ít dữ liệu nhất
        # eps là sai số chấp nhận, tức những lớp có số dữ liệu chênh với lớp MC trong phạm vi sai số này được cho vào fold
        # r là số dữ liệu nhiều nhất có thể, để được coi là xấp xỉ với lớp MC.
        # bou1 là những lớp xấp xỉ
        # bou2 là số lượng ảnh của các lớp trong bou1
        # MB là số lượng ảnh lớn nhất trong tập bou2
        F = []
        MC = np.where(B_temp > 0, B_temp, np.inf).argmin()
        eps = 0.5 * B_temp[MC]
        r = np.ceil(B_temp[MC]+eps)
        bou1 = np.where(B_temp <= r)[0]
        bou2 = B_temp[bou1]
        MB = B_temp[bou1[np.argmax(bou2)]]

        #Thu thập dữ liệu vào fold, Lấy ngẫu nhiên bằng cách trộn dữ liệu trong lớp đó bằng hàm shuffle, sau đó chọn nt
        # dữ liệu đầu tiên.
        for i in range (0, n_classes):
            # Nếu lớp i còn dữ liệu chua được chia vào fold
            if B_temp[i] > 0:
                nt = int(np.minimum(B_temp[i], MB))
                shuffle(A_temp[i])
                F += A_temp[i][:nt]
                del A_temp[i][:nt]
                B_temp[i] -= nt
            else:
                # Nếu các dữ liệu trong lớp i đã được chia hết vào fold thì ta khôi phục dữ liệu gốc của lớp này để chia
                shuffle(A[i])
                nt = int(np.minimum(B[i], MB))
                F += A[i][:nt]
        print(len(F))

        sumsum += len(F)
        total_fold.append(F)

    # Trả về tổng số fold và tổng số ảnh. Trường hợp bạn muốn coi dữ liệu từng fold thì bạn coi biến total_fold là xong.
    return len(total_fold), sumsum

def performModel(path2data, train_ori, val_ori, test_ori, class_encode, batch_train, batch_val, batch_test,
                 learning_rate, img_width, img_height):
    # Hàm này huấn luyện model theo YMufT. Trong bài báo https://www.mdpi.com/2076-3417/11/8/3331, tôi huấn luyện YMufT
    # sao cho tổng số lần đọc ảnh ở YMufT bằng hoặc nhỏ hơn số lần đọc ảnh ở cách huấn luyện thông thường (như cách huấn
    # luyện trong thư mục basic_generator. Xét tập dữ liệu trong thư mục Dataset, YMufT chia dữ liệu train thành 7 fold
    # (biến total_fold) với tổng số ảnh là training_step_ymuft. Khi đó, ở mỗi chu kỳ, tôi huấn luyện mỗi fold tầm n lần.
    # Tôi huấn luyện trong m chu kỳ. Trong bài báo trên, tôi huấn luyện trong num_loop_eps chu kỳ, mỗi fold ở mỗi chu kỳ
    # được huấn luyện num_loop_eps lần, và giảm dần qua từng chu kỳ. Ví dụ num_loop_eps = 4 thì ở chu kỳ 1, ta huấn luyện
    # mỗi fold 4 lần. Sang chu kỳ 2, mỗi fold được huấn luyện 3 lần, cho tới chu kỳ 4 thì mỗi fold huấn luyện 1 lần.
    # Bạn lưu ý rằng cách làm này là cách làm chủ quan của tác giả. Bạn có thể bỏ đi giới hạn số chu kỳ hay số vòng lặp,
    # chẳng hạn như huấn luyện fold 1 cho tới khi fold này hội tụ rồi mới qua fold 2, fold 3, ... và lặp lại chu kỳ cho
    # tới khi tất cả các fold đều hội tụ. Hoặc xài khái niệm Early stopping khu huấn luyện mỗi fold cũng được.

    # training_step_classical là số lần đọc ảnh ở cách huấn luyện thông thường. Nếu ở cách thông thường, ta huấn luyện
    # model trong 10 epoch thì training_step_classical bằng tích của tổng số ảnh huấn luyện và số epoch.
    training_step_classical = len(train_ori) * 10

    # Hàm collect_data_ymuft chính làm thuật toán YMufT chia dữ liệu tra các fold. Thực ra trong hàm YMufT_train.py (hàm
    # huấn luyện model) cũng có hàm tương tự collect_data_ymuft (self.YMufT()), nhưng hàm collect_data_ymuft dùng để tính
    # tổng số fold (total_fold) cũng như tổng số ảnh ở total_fold (training_step_ymuft). Từ đó, ta tính được số chu kỳ và
    # số vòng lặp (num_loop_eps) sao cho tổng số lần đọc ảnh ở YMufT bằng hoặc nhỏ hơn số lần đọc ảnh ở cách huấn luyện
    # thông thường
    total_fold, training_step_ymuft = collect_data_ymuft(train_ori, class_encode)
    num_loop_eps = np.int(np.floor((-1+np.sqrt(1+(8*training_step_classical/training_step_ymuft)))/2))

    # Bạn hãy coi file YMufT_train.py để hiểu cụ thể cách huấn luyện.
    chkPer = checkingPerformance(path2data = path2data, class_encode=class_encode, lr=learning_rate, img_width=img_width,
                                 img_height=img_height, img_channels=3, num_classes=len(class_encode))

    chkPer.training(train_ori=train_ori, batch_train=batch_train, val_ori=val_ori, batch_val=batch_val,
                    class_encode=class_encode, num_loop_eps=num_loop_eps, total_fold=total_fold, is_aug=True)

    chkPer.testing(test_ori=test_ori, batch_test=batch_test)

    del chkPer


def main_running():
    f = open('train_path.h5', 'rb')
    train_ori = pickle.load(f)
    f.close()

    f = open('val_path.h5', 'rb')
    val_ori = pickle.load(f)
    f.close()

    f = open('test_path.h5', 'rb')
    test_ori = pickle.load(f)
    f.close()

    f = open('class_encode.hdf5', 'rb')
    class_encode = pickle.load(f)
    f.close()

    performModel(path2data='Dataset', train_ori=train_ori, val_ori=val_ori, test_ori=test_ori, class_encode=class_encode,
                 batch_train=64, batch_val=43, batch_test=43, learning_rate=1e-3, img_width=128, img_height=128)
    gc.collect()

if __name__ == '__main__':
    path2data = 'Dataset'
    main_running()