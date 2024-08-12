import matplotlib.pyplot as plt
import pickle
import numpy as np
import random

def get_class_encode(path2data):
    # Mã hóa các lớp thành số nguyên dương để huấn luyện.
    # Input: path2data là đường dẫn tới các lớp trong bộ dữ liệu.
    # Output: mydict là từ điển chứa tên lớp và số nguyên mã hóa tương ứng.
    #         generator.filenames trả về list chuỗi tên hình có trong dữ liệu.
    # --------------------------------------------------------------------
    # Xài ImageDataGenerator trong Keras để mã hóa chuỗi lớp trong bộ dữ liệu
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen =  ImageDataGenerator()
    generator = datagen.flow_from_directory(path2data)

    # Kết quả Keras mã hóa chuỗi lớp thành số nguyên dương.
    # Dạng Dictionary. Nhập tên lớp, có số nguyên dương tương ứng.
    # Dùng để tạo one-hot encoder cũng như mấy chuyện sau này.
    mydict = generator.class_indices

    # Lưu lại từ điển này để xài sau này.
    f = open('class_encode.hdf5', 'wb')
    pickle.dump(mydict, f)
    f.close()

    return mydict, generator.filenames

def distribution_graph(lst_file, class_encode):
    # In hình phân bố ảnh trong dữ liệu
    # Input: lst_file là tập các đường dẫn tới tất cả các hình trong bộ dữ liệu.
    #        class_encode là từ điển mã hóa chuỗi lớp thành số.
    # Output: List 2 chiều, mỗi list con i chứa tất cả đường dẫn tới các ảnh trong lớp i.
    num_classes = len(class_encode)
    tke = np.zeros(num_classes, dtype=np.int)

    # Đếm thủ công số ảnh trong mỗi lớp.
    name = list(class_encode)
    lst_name_with_spe = [ [] for _ in range(num_classes) ]  #Tạo list rỗng, có n list rỗng con,
                                                            # với n là tổng số lớp trong bộ dữ liệu
    for name_file in lst_file:
        flag = name_file.find('\\')
        name_class = name_file[:flag]
        tke[class_encode[name_class]] += 1 # Nhập chuỗi 'name_class' vào dictionary class_encode để có số mã hóa tương
                                           # ứng. Xem lại hàm get_class_encode
        lst_name_with_spe[class_encode[name_class]].append(name_file[flag+1:])

    plt.bar(name, list(tke))
    plt.xlabel('Class')
    plt.ylabel('Number of images')

    plt.xticks(rotation=90)
    plt.title('Distribution of images')
    plt.tight_layout()
    plt.savefig('data_distribtion.png', bbox_inches='tight', dpi = 500)
    plt.clf()
    plt.close()

    # lst_name_with_spe lúc này có n list con, mỗi list i chứa tất cả đường dẫn tới các ảnh trong lớp i.
    # Xài list này để chia dữ liệu trong hàm data_division.
    return lst_name_with_spe

def check_path(dt, class_name):
    # Hàm này biến chuỗi (string) đường dẫn vào hình trở thành 1 từ điển (dictionary).
    # Ta sẽ xài từ điển này để đọc dữ liệu khi huấn luyện model.
    # Thực ra, việc đọc dữ liệu chỉ cần đường dẫn kiểu chuỗi là đủ. Nhưng nếu bạn muốn huấn luyện theo YMufT thì bạn
    # hãy xài từ điển.
    # Input: dt là list chỉ chứa tên của hình, ví dụ abc.jpg
    #        class_name là tên lớp
    # Output: dat là thư viện dành cho lớp class_encode
    dat = []

    for k in dt:
        # Ví dụ: 'flower_photos\daisy\5547758_eea9edfd54_n.jpg'
        # Từ điển biến chuỗi này thành
        #   'class': daisy
        #   'name_img': 5547758_eea9edfd54_n.jpg
        ID = {'class': class_name, 'name_img': k}
        dat.append(ID)

    return dat

def data_division(lst_data, class_encode):
    # Chia dữ liệu làm 3 tập train, val, test
    # Ở mỗi lớp, ta trộn dữ liệu, xong chia dữ liệu thành 3 tâp với tỉ lệ 6:2:2
    # Input: lst_data: List 2 chiều, mỗi list con i chứa tất cả tên hình trong lớp i.
    #        class_encode là từ điển mã hóa tên lớp
    # Chạy hàm này, ta được 3 tập tin train_path.h5, val_path.h5 và test_path.h5. Mỗi tập tin là 1 list chứa toàn bộ
    # đường dẫn (dưới dạng từ điển) tới hình tương ứng.
    train_path = []
    val_path = []
    test_path = []

    for i, dt in enumerate(lst_data):
        # Trộn dữ liệu
        random.shuffle(dt)
        random.shuffle(dt)
        random.shuffle(dt)

        data_len = len(dt)
        num_test = int(0.2 * data_len)
        num_val = int(0.2 * data_len)
        num_train = data_len - num_test - num_val

        dtrain = check_path(dt[:num_train], list(class_encode)[i])
        dval   = check_path(dt[num_train:num_train+num_val], list(class_encode)[i])
        dtest  = check_path(dt[num_train+num_val:], list(class_encode)[i])

        train_path += dtrain
        val_path += dval
        test_path += dtest

    f = open('train_path.h5', 'wb')
    pickle.dump(train_path, f)
    f.close()

    f = open('val_path.h5', 'wb')
    pickle.dump(val_path, f)
    f.close()

    f = open('test_path.h5', 'wb')
    pickle.dump(test_path, f)
    f.close()

if __name__ == '__main__':
    class_encode, list_files = get_class_encode(path2data = 'Dataset')
    lst_name_with_spe = distribution_graph(lst_file=list_files, class_encode=class_encode)
    data_division(lst_data = lst_name_with_spe, class_encode = class_encode)