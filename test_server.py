import socket
import time
import os

# Server configuration
HOST = '127.0.0.1'
PORT = 8888
DATA_FILE = 'stream_data.bin'
CHUNK_SIZE = 4096 # Read and send in 4KB chunks

def run_server():
    """
    A simple TCP server that sends the contents of a binary file to
    any connecting client.
    """
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found.")
        print("Please generate it first by running 'generate_stream_data.py'")
        return

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Reuse address to avoid "Address already in use" errors on quick restarts
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((HOST, PORT))
    except OSError as e:
        print(f"Error binding to {HOST}:{PORT}: {e}")
        print("Is another process already using this port?")
        return

    server_socket.listen()
    print(f"Test server listening on {HOST}:{PORT}")
    print(f"Will send '{DATA_FILE}' to connecting clients.")

    try:
        while True:
            conn, addr = server_socket.accept()
            with conn:
                print(f"\nConnected by {addr}")
                try:
                    # Send the 2-frame data file enough times to make up 59 frames
                    # (59+1)/2 = 30 times
                    print("Sending 59 frames worth of data (by repeating the 2-frame file)...")
                    total_sent = 0
                    for _ in range(30):
                        with open(DATA_FILE, 'rb') as f:
                            while True:
                                chunk = f.read(CHUNK_SIZE)
                                if not chunk:
                                    break # End of file
                                conn.sendall(chunk)
                                total_sent += len(chunk)
                    print(f"Successfully sent {total_sent} bytes.")
                except IOError as e:
                    print(f"Error reading data file: {e}")
                except socket.error as e:
                    print(f"Socket error during send: {e}")
                finally:
                    print("Client disconnected.")
    except KeyboardInterrupt:
        print("\nServer is shutting down.")
    finally:
        server_socket.close()

if __name__ == '__main__':
    run_server()
