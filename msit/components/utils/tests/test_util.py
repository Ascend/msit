import unittest
from unittest.mock import patch
from itertools import product

from components.utils.util import (confirmation_interaction, 
                                   check_file_ext, 
                                   check_file_size_based_on_ext)


class TestUtil(unittest.TestCase):
    
    def test_confirmation_interaction_yes(self):
        yes_input = ['y', 'Y', 'yes', 'YES', 'Yes', 'yES']
        
        for i in yes_input:
            with self.subTest(i):
                with patch('builtins.input', return_value=i):
                    self.assertTrue(confirmation_interaction(""))
    
    def test_confirmation_interaction_no(self):
        no_input = ['n', 'no', 'abc', 'EOF']
        
        for i in no_input:
            with self.subTest(i):
                with patch('builtins.input', return_value=i):
                    self.assertFalse(confirmation_interaction(""))
                    
    def test_check_file_ext_type_error(self):
        paths = [1, 2.5, -3+1j]
        exts = [range(10), set(), dict()]
        
        for path, ext in product(paths, exts):
            with self.subTest(path=path, ext=ext):
                self.assertRaises(TypeError, check_file_ext, path, ext)
    
    def test_check_file_ext_not_match(self):
        path = 'model.onnx'
        exts = ['onnx', 'py', '.py', '.cpp']
        
        for ext in exts:
            with self.subTest(path=path, ext=ext):
                self.assertFalse(check_file_ext(path, ext))
    
    def test_check_file_ext_not_match(self):
        paths = ['model.onnx', 'test.py', 'main.cpp']
        exts = ['.onnx', '.py', '.cpp']
        
        for path, ext in zip(paths, exts):
            with self.subTest(path=path, ext=ext):
                self.assertTrue(check_file_ext(path, ext))
    
    def test_check_file_size_based_on_ext_type_error(self):
        paths = [1, 2.5, -3+1j]
        
        for path in paths:
            with self.subTest(path=path):
                self.assertRaises(TypeError, check_file_size_based_on_ext, path)
                
    def test_check_file_size_based_on_ext_large_size(self):
        exts = ['.csv', '.json', '.txt', '.onnx', '.ini', '.py', '.pth', '.bin']
        
        with patch('os.path.getsize', return_value=500 * 1024 * 1024 * 1024):
            for ext in exts:
                with self.subTest(ext=ext):
                    self.assertFalse(check_file_size_based_on_ext('random_file', ext))
                    self.assertFalse(check_file_size_based_on_ext('random_file' + ext))
        
            with patch('builtins.input', return_value='n'):
                self.assertFalse(check_file_size_based_on_ext('random_file'))
        
    
    def test_check_file_size_based_on_ext_normal_size(self):
        config_file_size = 8 * 1024
        text_file_size = 8 * 1024 * 1024
        onnx_model_size = 1 * 1024 * 1024 * 1024
        model_weigtht_size = 8 * 1024 * 1024 * 1024
        
        exts = ['.ini', '.csv', '.json', '.txt', '.py', '.onnx', '.pth', '.bin']
        sizes = [config_file_size] + [text_file_size] * 4 + [onnx_model_size] + [model_weigtht_size] * 2
        
        for ext, size in zip(exts, sizes): 
            with patch('os.path.getsize', return_value=size):
                with self.subTest(ext=ext, size=size):
                    self.assertTrue(check_file_size_based_on_ext('random_file', ext))
                    self.assertTrue(check_file_size_based_on_ext('random_file' + ext))
                    self.assertTrue(check_file_size_based_on_ext('random_file'))
