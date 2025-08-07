# 你可以把这个类放在一个单独的 utils.py 文件中
import json
import os
import hashlib
from filelock import FileLock, Timeout  # 需要安装: pip install filelock


class JsonFileStateCache:
    def __init__(self, session_id: str, cache_dir: str = ".cache"):
        self.session_id = session_id
        self.cache_dir = cache_dir
        # 每个会话对应一个独立的 JSON 文件
        self.cache_file_path = os.path.join(self.cache_dir, f"{self.session_id}.json")
        self.lock_file_path = f"{self.cache_file_path}.lock"

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _create_task_signature(function_name: str, args: dict) -> str:
        """根据函数名和参数创建一个唯一的、确定的任务签名。"""
        sorted_args_str = json.dumps(args, sort_keys=True)
        full_signature_str = f"{function_name}:{sorted_args_str}"
        return hashlib.sha256(full_signature_str.encode()).hexdigest()

    def _read_cache(self) -> dict:
        """从 JSON 文件安全地读取数据。"""
        if not os.path.exists(self.cache_file_path):
            return {}
        try:
            with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # 如果文件损坏或为空，返回空字典
            return {}

    def _write_cache(self, data: dict):
        """将数据安全地写入 JSON 文件。"""
        with open(self.cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get(self, function_name: str, args: dict) -> str | None:
        """根据任务签名从缓存中获取结果。"""
        task_key = self._create_task_signature(function_name, args)

        # 使用文件锁确保读取时文件不被其他进程修改
        with FileLock(self.lock_file_path, timeout=5):
            cache_data = self._read_cache()
            return cache_data.get(task_key)

    def set(self, function_name: str, args: dict, result: str):
        """将任务结果存入缓存。"""
        task_key = self._create_task_signature(function_name, args)

        # 使用文件锁来处理并发写入，防止数据损坏
        with FileLock(self.lock_file_path, timeout=5):
            cache_data = self._read_cache()
            cache_data[task_key] = result
            self._write_cache(cache_data)

    def clear(self):
        """清空当前会话的所有缓存。"""
        with FileLock(self.lock_file_path, timeout=5):
            if os.path.exists(self.cache_file_path):
                os.remove(self.cache_file_path)