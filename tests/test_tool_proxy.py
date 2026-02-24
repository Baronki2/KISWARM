import unittest
from flask import json
from your_flask_app import app  # Make sure to replace this with actual import

class TestToolProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = app.test_client()
        cls.app.testing = True

    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'status': 'healthy'})

    def test_list_tools(self):
        response = self.app.get('/tools')
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json, list)

    def test_execute_tool(self):
        response = self.app.post('/execute', json={'tool_id': 'tool1', 'params': {}})
        self.assertEqual(response.status_code, 200)
        self.assertIn('result', response.json)

    def test_register_tool(self):
        response = self.app.post('/register', json={'tool_id': 'tool2', 'tool_data': {}})
        self.assertEqual(response.status_code, 201)
        self.assertIn('tool_id', response.json)

    def test_status_tool(self):
        response = self.app.get('/status/tool1')
        self.assertEqual(response.status_code, 200)
        self.assertIn('status', response.json)

    def test_error_handling(self):
        response = self.app.get('/error')  # assume '/error' is an endpoint that raises an error
        self.assertEqual(response.status_code, 500)
        self.assertIn('error', response.json)

    # Edge case tests
    def test_execute_tool_empty_params(self):
        response = self.app.post('/execute', json={'tool_id': 'tool1'})
        self.assertEqual(response.status_code, 400)

    def test_register_tool_invalid_data(self):
        response = self.app.post('/register', json={'tool_id': 'tool2'})  # Missing 'tool_data'
        self.assertEqual(response.status_code, 400)

    # Security tests
    def test_execute_tool_security(self):
        response = self.app.post('/execute', json={'tool_id': 'tool1', 'params': {'malicious': '<script>'}})
        self.assertEqual(response.status_code, 400)  # expecting a validation error

if __name__ == '__main__':
    unittest.main()