"""
1. 拷贝多个request batch到一个目录中，重命为need的格式
"""
import shutil
from pathlib import Path

batch_request_dir = r"D:\下载D\deepseek\collect_forward_info_3"

new_dir = Path(r"D:\PyProject\ModelEvalState\data\v1.0.0")
middle_prefix = "deepseek_r1_yunneng_forward"

batch_request_files = {}
all_pid = set()
for child in Path(batch_request_dir).iterdir():
    name_list = child.name.split("_")
    file_name, pid = name_list[:-1], name_list[-1]
    if pid in batch_request_files:
        batch_request_files[pid].append(child)
    else:
        batch_request_files[pid] = [child]
    all_pid.add(pid)

# deep_seek_base = Path(r"D:\PyProject\ModelEvalState\data\v1.0.0\deep_seek_r1_base")

for i in range(len(batch_request_files)):
    middle_path = new_dir.joinpath(f{middle_prefix}_{i}")
    middle_path.mkdir()
    for child in batch_request_files[all_pid.pop()]:
        name_list = child.stem.split("_")
        file_name, pid = "_".join(name_list[:-1]), name_list[-1]
        shutil.copy(child, middle_path.joinpath(f"{file_name}_{i}.csv"))
    # for child in deep_seek_base.iterdir():
    #       shutil.copy(child, middle_path.joinpath(child.name))

