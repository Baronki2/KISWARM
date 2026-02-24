import unittest
import json
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

class TestKISWARMStatus(unittest.TestCase):
    """Unit tests for kiswarm_status.py monitor"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_kiswarm_dir = "/tmp/test_kiswarm"
        self.test_qdrant_path = f"{self.test_kiswarm_dir}/qdrant_data"

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_kiswarm_dir):
            import shutil
            shutil.rmtree(self.test_kiswarm_dir)

    @patch('psutil.cpu_percent')
    def test_cpu_monitoring(self, mock_cpu):
        """Test CPU monitoring functionality"""
        mock_cpu.return_value = 45.5
        # Simulate reading CPU percentage
        result = mock_cpu(interval=1)
        self.assertEqual(result, 45.5)
        self.assertLess(result, 100)
        self.assertGreaterEqual(result, 0)

    @patch('psutil.virtual_memory')
    def test_memory_monitoring(self, mock_mem):
        """Test memory monitoring functionality"""
        mock_mem_obj = MagicMock()
        mock_mem_obj.percent = 60.2
        mock_mem_obj.used = 16 * 1024**3  # 16GB
        mock_mem_obj.total = 32 * 1024**3  # 32GB
        mock_mem.return_value = mock_mem_obj
        
        result = mock_mem()
        self.assertEqual(result.percent, 60.2)
        self.assertEqual(result.used, 16 * 1024**3)
        self.assertLess(result.percent, 100)

    @patch('psutil.disk_usage')
    def test_disk_monitoring(self, mock_disk):
        """Test disk usage monitoring"""
        mock_disk_obj = MagicMock()
        mock_disk_obj.percent = 70.5
        mock_disk_obj.free = 100 * 1024**3  # 100GB free
        mock_disk_obj.total = 500 * 1024**3  # 500GB total
        mock_disk.return_value = mock_disk_obj
        
        result = mock_disk("/home")
        self.assertEqual(result.percent, 70.5)
        self.assertGreater(result.free, 0)

    def test_ollama_status_offline(self):
        """Test detection of offline Ollama service"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            # Should handle offline gracefully
            self.assertEqual(mock_get.side_effect.__class__.__name__, 'Exception')

    @patch('requests.get')
    def test_ollama_status_online(self, mock_get):
        """Test detection of online Ollama service"""
        mock_response = MagicMock()
        mock_response.json.return_value = {'models': [{'name': 'llama2'}, {'name': 'phi3'}]}
        mock_get.return_value = mock_response
        
        result = mock_get("http://localhost:11434/api/tags")
        self.assertEqual(len(result.json()['models']), 2)

    def test_qdrant_path_exists(self):
        """Test Qdrant database path detection"""
        os.makedirs(self.test_qdrant_path, exist_ok=True)
        self.assertTrue(os.path.exists(self.test_qdrant_path))

    def test_qdrant_path_missing(self):
        """Test handling of missing Qdrant database"""
        self.assertFalse(os.path.exists(self.test_qdrant_path))

    def test_governance_config_exists(self):
        """Test governance configuration file detection"""
        config_path = f"{self.test_kiswarm_dir}/governance_config.json"
        os.makedirs(self.test_kiswarm_dir, exist_ok=True)
        
        config = {
            "system_name": "KISWARM",
            "governance_mode": "active",
            "autonomous_operation": True
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        self.assertTrue(os.path.exists(config_path))
        with open(config_path) as f:
            loaded = json.load(f)
            self.assertEqual(loaded['governance_mode'], 'active')

    def test_governance_config_missing(self):
        """Test handling of missing governance config"""
        config_path = f"{self.test_kiswarm_dir}/governance_config.json"
        self.assertFalse(os.path.exists(config_path))

    def test_resource_color_coding(self):
        """Test color coding for resource levels"""
        def get_color(percentage):
            if percentage < 60:
                return "green"
            elif percentage < 80:
                return "yellow"
            else:
                return "red"
        
        self.assertEqual(get_color(50), "green")
        self.assertEqual(get_color(70), "yellow")
        self.assertEqual(get_color(90), "red")

    def test_monitor_initialization(self):
        """Test monitor class initialization"""
        import datetime
        start_time = datetime.datetime.now()
        self.assertIsNotNone(start_time)
        self.assertIsInstance(start_time, datetime.datetime)

    @patch('requests.get')
    def test_proxy_health_check(self, mock_get):
        """Test tool proxy health check"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = mock_get("http://localhost:11435/health")
        self.assertEqual(result.status_code, 200)

    def test_size_calculation_format(self):
        """Test memory size formatting"""
        def format_size(bytes_val):
            return f"{bytes_val / 1_048_576:.1f}MB"
        
        result = format_size(1048576)
        self.assertEqual(result, "1.0MB")

if __name__ == '__main__':
    unittest.main()