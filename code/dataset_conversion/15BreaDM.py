# import os

# def create_classification_file(base_dir, output_file):
#     """
#     创建分类文本文件，包含文件路径和标签
    
#     参数:
#     base_dir: 基础目录，包含Benign和Malignant文件夹
#     output_file: 输出文本文件路径
#     """
#     lines = []
    
#     # 处理Benign文件夹 (标签0)
#     benign_dir = os.path.join(base_dir, 'Benign')
#     if os.path.exists(benign_dir):
#         for case_dir in sorted(os.listdir(benign_dir)):
#             case_path = os.path.join(benign_dir, case_dir)
#             if os.path.isdir(case_path):
#                 # 添加pre-dce和pos-dce文件的路径和标签
#                 pre_dce_path = os.path.join('Benign', case_dir)
#                 # pos_dce_path = os.path.join('Benign', case_dir)
#                 lines.append(f"{pre_dce_path} 0")
#                 # lines.append(f"{pos_dce_path} 0")
    
#     # 处理Malignant文件夹 (标签1)
#     malignant_dir = os.path.join(base_dir, 'Malignant')
#     if os.path.exists(malignant_dir):
#         for case_dir in sorted(os.listdir(malignant_dir)):
#             case_path = os.path.join(malignant_dir, case_dir)
#             if os.path.isdir(case_path):
#                 # 添加pre-dce和pos-dce文件的路径和标签
#                 pre_dce_path = os.path.join('Malignant', case_dir)
#                 # pos_dce_path = os.path.join('Malignant', case_dir)
#                 lines.append(f"{pre_dce_path} 1")
#                 # lines.append(f"{pos_dce_path} 1")
    
#     # 写入文本文件
#     with open(output_file, 'w') as f:
#         f.write('\n'.join(lines))
    
#     print(f"分类文件已创建: {output_file}")
#     print(f"总共 {len(lines)} 个条目")

# # 主程序
# if __name__ == "__main__":
#     # 基础目录(包含Benign和Malignant文件夹的train目录)
#     base_dir = r'/data/zzn/UniMRINet/dataset/MR_Dataset/15BreaDM/test'
    
#     # 输出文本文件路径
#     output_file = r'/data/zzn/UniMRINet/dataset/dataset_orginal/txt/15BreaDM/test.txt'
    
#     # 创建分类文件
#     create_classification_file(base_dir, output_file)

import os
print(len(os.listdir('/data/zzn/UniMRINet/dataset/MR_Dataset/15BreaDM')))