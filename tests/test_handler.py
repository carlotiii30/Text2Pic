import os
import unittest
from unittest.mock import MagicMock

from src.handler import Handler


class TestHandler(unittest.TestCase):
    def test_execute_command_generate_number_success(self):
        # Arrange
        request = {
            "command": "generate_number",
            "text": "5"
        }
        expected_file_path = "data/nums/drawn_number_5.png"
        os.makedirs(os.path.dirname(expected_file_path), exist_ok=True)
        handler = Handler(MagicMock())

        # Act
        handler._execute_command(request)

        # Assert
        self.assertTrue(os.path.exists(expected_file_path))

        # Cleanup
        os.remove(expected_file_path)

    def test_execute_command_generate_number_error(self):
        # Arrange
        request = {
            "command": "generate_number",
            "text": "abc"
        }
        expected_response = {
            "status": "error",
            "message": "Error generating the image: invalid literal for int() with base 10: 'abc'"
        }
        handler = Handler(MagicMock())

        # Act
        response = handler._execute_command(request)

        # Assert
        self.assertEqual(response, expected_response)

    def test_execute_command_unknown_command(self):
        # Arrange
        request = {
            "command": "unknown_command",
            "text": "5"
        }
        expected_response = {
            "status": "error",
            "message": "Unknown command: unknown_command"
        }
        handler = Handler(MagicMock())

        # Act
        response = handler._execute_command(request)

        # Assert
        self.assertEqual(response, expected_response)

if __name__ == '__main__':
    unittest.main()