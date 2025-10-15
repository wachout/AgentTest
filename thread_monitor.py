import threading
import time
from datetime import datetime

class ThreadMonitor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.thread_data = {}
            self.lock = threading.Lock()
            self.initialized = True

    def thread_started(self, thread_name):
        with self.lock:
            start_time = time.time()
            self.thread_data[thread_name] = {
                'start_time': start_time,
                'start_time_str': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': None,
                'end_time_str': None,
                'duration': None
            }
        print(f"Thread {thread_name} started at {self.thread_data[thread_name]['start_time_str']}")

    def thread_finished(self, thread_name):
        with self.lock:
            if thread_name in self.thread_data and self.thread_data[thread_name].get('start_time') is not None:
                end_time = time.time()
                start_time = self.thread_data[thread_name]['start_time']
                duration = end_time - start_time
                self.thread_data[thread_name].update({
                    'end_time': end_time,
                    'end_time_str': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': duration
                })
                print(f"Thread {thread_name} finished at {self.thread_data[thread_name]['end_time_str']}. Duration: {duration:.2f} seconds.")

    def get_report(self):
        report = "--- Thread Monitoring Report ---\n"
        with self.lock:
            for name, data in self.thread_data.items():
                report += f"Thread: {name}\n"
                report += f"  Start Time: {data['start_time_str']}\n"
                if data['end_time_str']:
                    report += f"  End Time:   {data['end_time_str']}\n"
                    report += f"  Duration:   {data['duration']:.4f} seconds\n"
                else:
                    report += "  Status:     Running\n"
                report += "-" * 20 + "\n"
        return report