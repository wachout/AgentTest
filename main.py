import threading
import time
import random
from thread_monitor import ThreadMonitor

def worker(thread_name):
    """A function that simulates work for a thread."""
    monitor = ThreadMonitor()
    monitor.thread_started(thread_name)

    # Simulate some work
    sleep_time = random.uniform(1, 3)
    time.sleep(sleep_time)

    monitor.thread_finished(thread_name)

if __name__ == "__main__":
    threads = []
    thread_count = 5

    print(f"Starting {thread_count} worker threads...")

    # Create and start threads
    for i in range(thread_count):
        thread_name = f"Worker-{i+1}"
        thread = threading.Thread(target=worker, args=(thread_name,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("\nAll threads have completed their execution.")

    # Get the final report
    monitor = ThreadMonitor()
    report = monitor.get_report()
    print(report)