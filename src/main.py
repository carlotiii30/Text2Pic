from src import server

if __name__ == "__main__":
    server = server.Server("localhost", 12345)
    server.start()
