import json
import logging

from src import drawing, utils


# pylint: disable=too-few-public-methods
class Handler:
    """Class that handles client requests.

    This class handles the requests sent by the client and executes the
    corresponding actions.

    Attributes:
        socket (socket): Client socket.
    """

    def __init__(self, socket):
        """Class constructor.

        Args:
            socket: Client socket.
        """
        self.socket = socket

    def handle(self):
        """Method that handles the client's request."""
        with self.socket:
            data = self._receive_data()
            request = self._process_request(data)
            response = self._execute_command(request)
            self._send_response(response)

    def _receive_data(self):
        """Method that receives the data sent by the client.

        Returns:
            str: Data sent by the client.
        """
        return self.socket.recv(1024).decode()

    def _process_request(self, data):
        """Method that processes the data received from the client.

        Args:
            data (str): Data sent by the client.

        Returns:
            dict: Request data.
        """
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
        """Method that executes the command sent by the client.

        Args:
            request (dict): Request data.

        Returns:
            dict: Response data.
        """
        command = request.get("command")
        text = request.get("text", "")

        if command == "generate_number":
            response = self._generate_number(text)
        else:
            response = {
                "status": "error",
                "message": f"Unknown command: {command}",
            }

            logging.error("Unknown command: %s", command)

        return response

    def _generate_number(self, text):
        """Method that generates an image of a number.

        Args:
            text (str): Number to generate.

        Returns:
            dict: Response data.
        """
        try:
            cond_gan = utils.load_model_with_weights("models/cgan_nums.weights.h5")
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

    def _send_response(self, response):
        """Method that sends the response to the client.

        Args:
            response (dict): Response data.

        Returns:
            dict: Response data.
        """
        self.socket.sendall(json.dumps(response).encode())
