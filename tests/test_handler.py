import os
import unittest
from unittest.mock import MagicMock

from src.handler import Handler


class TestHandler(unittest.TestCase):
    def test_handle_receive_data(self):
        socket_mock = MagicMock()
        socket_mock.recv.return_value = b'{"command": "generate_number", "text": "5"}'
        handler = Handler(socket_mock)

        data = handler._receive_data()

        self.assertEqual(data, '{"command": "generate_number", "text": "5"}')

    def test_process_request_valid_json(self):
        handler = Handler(MagicMock())
        data = '{"command": "generate_number", "text": "5"}'

        request = handler._process_request(data)

        self.assertEqual(request, {"command": "generate_number", "text": "5"})

    def test_process_request_invalid_json(self):
        handler = Handler(MagicMock())
        data = '{"command": "generate_number", "text": "5"'

        request = handler._process_request(data)

        self.assertEqual(request["status"], "error")
        self.assertIn("Error decoding JSON", request["message"])

    def test_execute_command_generate_number_success(self):
        request = {"command": "generate_number", "text": "5"}

        expected_file_path = "data/nums/drawn_number_5.png"
        os.makedirs(os.path.dirname(expected_file_path), exist_ok=True)
        handler = Handler(MagicMock())

        handler._execute_command(request)
        self.assertTrue(os.path.exists(expected_file_path))
        os.remove(expected_file_path)

    def test_execute_command_generate_number_error(self):
        request = {"command": "generate_number", "text": "abc"}
        expected_response = {
            "status": "error",
            "message": "Error generating the image: invalid literal for int() with base 10: 'abc'",
        }
        handler = Handler(MagicMock())

        response = handler._execute_command(request)
        self.assertEqual(response, expected_response)

    def test_execute_command_unknown_command(self):
        request = {"command": "unknown_command", "text": "5"}
        expected_response = {"status": "error", "message": "Unknown command: unknown_command"}
        handler = Handler(MagicMock())

        response = handler._execute_command(request)
        self.assertEqual(response, expected_response)

    def test_send_response(self):
        socket_mock = MagicMock()
        handler = Handler(socket_mock)
        response = {"status": "success", "message": "Image generated successfully"}

        handler._send_response(response)

        socket_mock.sendall.assert_called_once_with(b'{"status": "success", "message": "Image generated successfully"}')
