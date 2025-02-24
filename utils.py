from typing import List, Dict
import os
import shutil
def file_move(file_path: str, target_list: list, image_path: str, dst_path: str) -> (List, List):
    """
    将文件中的图像根据标签移动到指定的目标文件夹中

    Args:
        file_path (str): 文件路径
        target_list (list): 目标标签列表
        image_path (str): 图像文件夹路径
        dst_path (str): 目标文件夹路径

    Returns:
        List: 移动成功的标签列表
        List: 移动失败的标签列表
    """
    label_list = []
    image_list = []

    # 读取文件进行处理
    with open(file_path, 'r') as f:
        line = f.readline()
        line_split = line.split()

        label = line_split[-1]
        image = line_split[0]

        if int(label) in target_list:
            img_path = os.path.join(image_path, image)
            shutil.copy(img_path, os.path.join(dst_path,str(label)))

        while line:
            line = f.readline()
            if line:
                line_split = line.split()
                label = line_split[1]
                image = line_split[0]
                if int(label) in target_list:
                    img_path = os.path.join(image_path, image)
                    shutil.copy(img_path, os.path.join(dst_path, str(label)))

    make_dirs(target_list=target_list, move_path="./val")
    # make_dirs(target_list=target_list,move_path="./train")
    print("over")



def make_dirs(target_list, move_path: str):
    # 创建文件夹
    for i in target_list:
        if isinstance(i, int):
            if not os.path.exists(os.path.join(move_path, str(i))):
                os.makedirs(os.path.join(move_path, str(i)))
        else:
            if not os.path.exists(os.path.join(move_path, i)):
                os.makedirs(os.path.join(move_path, i))


def class2index(classes_name: list, categories_file="./test.txt") -> (List, Dict):
    # 将类名转换为索引
    class2index_dict = {}
    index_list = []
    with open(categories_file, 'r') as f:
        line = f.readline()
        line_split = line.split()
        index = line_split[-1]
        name = line_split[0]
        # 将类名和索引成为字典
        class2index_dict[name] = index
        while line:
            line = f.readline()
            if line:
                line_split = line.split()
                index = line_split[-1]
                name = line_split[0]
            class2index_dict[name] = index
    # 将对应的转换为序列
    for i in classes_name:
        i = "/" + i[0] + "/" + i
        index = class2index_dict.get(i)
        if index is not None:
            index_list.append(int(index))
    return index_list, class2index_dict


if __name__ == '__main__':
    CLASSES_NAME = [
        "1", "2", "3", "4", "5", "6"
    ]
    make_dirs(CLASSES_NAME, './val/')
    make_dirs(CLASSES_NAME, './train/')
    index_list, class2index_dict = class2index(CLASSES_NAME)
    file_move('./test.txt',
            target_list=index_list,
            image_path="./val/val_256",
            dst_path="./val/")
