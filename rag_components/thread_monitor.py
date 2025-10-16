import time
import threading
from typing import Dict, Any

class ThreadMonitor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.thread_data = {}
                cls._instance.data_lock = threading.Lock()
        return cls._instance

    def start(self, thread_name: str):
        with self.data_lock:
            self.thread_data[thread_name] = {
                "start_time": time.time(),
                "end_time": None,
                "duration": None,
            }

    def stop(self, thread_name: str):
        with self.data_lock:
            if thread_name in self.thread_data:
                end_time = time.time()
                self.thread_data[thread_name]["end_time"] = end_time
                self.thread_data[thread_name]["duration"] = (
                    end_time - self.thread_data[thread_name]["start_time"]
                )

    def get_report(self) -> Dict[str, Any]:
        with self.data_lock:
            return self.thread_data.copy()

# Global instance
thread_monitor = ThreadMonitor()