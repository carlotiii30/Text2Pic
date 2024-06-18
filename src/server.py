import socket
import threading

from src.handler import Handler


class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen()
            print(f"Server listening on port {self.port}")
            while True:
                client_socket, _ = server_socket.accept()
                threading.Thread(target=self.client_handler, args=(client_socket,)).start()

    def client_handler(self, client_socket):
        handler = Handler(client_socket)
        handler.handle()
