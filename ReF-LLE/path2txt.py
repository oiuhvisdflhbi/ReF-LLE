import os

def write_file_paths_to_txt(folder_path, output_file):

    file_paths = [os.path.abspath(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


    prefix = "/home/zm/ReLLIE-copy/"
    file_paths = [path.replace(prefix, "") for path in file_paths]


    with open(output_file, 'w') as f:
        for path in file_paths:
            f.write(path + '\n')


folder_path = './reference/huawei_ref/'
output_file = './data/huawei_ref.txt'

write_file_paths_to_txt(folder_path, output_file)
