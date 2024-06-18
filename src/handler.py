import json
import logging

from src import drawing, utils
from src.nums import builders


# pylint: disable=too-few-public-methods
class Handler:
    def __init__(self, socket):
        self.socket = socket

    def handle(self):
        with self.socket:
            data = self._receive_data()
            request = self._process_request(data)
            response = self._execute_command(request)
            self._send_response(response)

    def _receive_data(self):
        return self.socket.recv(1024).decode()

    def _process_request(self, data):
        try:
            return json.loads(data)

        except json.JSONDecodeError as e:
            response = {
                "status": "error",
                "message": f"Error decoding JSON: {str(e)}",
            }

            logging.error("Error decoding JSON: %s", str(e))

            return response

    def _execute_command(self, request):
        command = request.get("command")
        text = request.get("text", "")

        if command == "generate_number":
            response = self._generate_number(text)

        elif command == "generate_image":
            response = {
                "status": "error",
                "message": "Command not implemented",
            }

            logging.error("Command not implemented: %s", command)

        else:
            response = {
                "status": "error",
                "message": f"Unknown command: {command}",
            }

            logging.error("Unknown command: %s", command)

        return response

    def _generate_number(self, text):
        try:
            generator, discriminator = builders.build_models()
            cond_gan = builders.build_conditional_gan(generator, discriminator)

            cond_gan = utils.load_model_with_weights("models/cgan_nums.weights.h5", cond_gan)

        except FileNotFoundError as e:
            response = {
                "status": "error",
                "message": f"Model file not found: {str(e)}",
            }
            logging.error("Model file not found: %s", str(e))

            return response

        try:
            img = drawing.draw_number(text, cond_gan)
            img = img.tolist()

            response = {
                "status": "success",
                "message": "Image generated successfully",
                "image": img,
            }

            logging.info("Image generated successfully: %s", text)

            return response

        except Exception as e:
            response = {
                "status": "error",
                "message": f"Error generating the image: {str(e)}",
            }

            logging.error("Error generating the image: %s", str(e))

            return response

    def _generate_image(self, text):
        # TODO: Implement this method
        pass

    def _send_response(self, response):
        self.socket.sendall(json.dumps(response).encode())
