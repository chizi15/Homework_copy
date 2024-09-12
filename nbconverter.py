import os
import nbformat
from nbconvert import PythonExporter


def convert_ipynb_to_py(ipynb_path, py_path):
    # 读取 .ipynb 文件
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # 使用 PythonExporter 将 .ipynb 转换为 .py
    python_exporter = PythonExporter()
    python_code, _ = python_exporter.from_notebook_node(notebook)

    # 将转换后的代码写入 .py 文件
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(python_code)


def batch_convert_ipynb_to_py(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".ipynb"):
            ipynb_path = os.path.join(directory, filename)
            py_path = os.path.join(directory, filename.replace(".ipynb", ".py"))
            convert_ipynb_to_py(ipynb_path, py_path)
            print(f"Converted\n\n {ipynb_path}\n\n to \n\n{py_path}")


# 指定要转换的目录
directory = r"D:\Work info\Study\AI\lihongyi\leedl-tutorial-1.2.1\Homework_copy"
for foldername in os.listdir(directory):
    if os.path.isdir(os.path.join(directory, foldername)):
        directory_each = os.path.join(directory, foldername)
        batch_convert_ipynb_to_py(directory_each)
    elif foldername.endswith(".ipynb"):
        ipynb_path = os.path.join(directory, foldername)
        py_path = os.path.join(directory, foldername.replace(".ipynb", ".py"))
        convert_ipynb_to_py(ipynb_path, py_path)
        print(f"Converted {ipynb_path} to {py_path}")
    else:
        print(f"Error: {foldername} is not a directory or .ipynb file")
