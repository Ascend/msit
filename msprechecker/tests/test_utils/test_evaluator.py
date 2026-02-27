# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import unittest

from msprechecker.utils import Evaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator

    def test_addition(self):
        self.assertEqual(self.evaluator.evaluate("2 + 3"), 5)
        self.assertEqual(self.evaluator.evaluate("-2 + 3"), 1)
        self.assertEqual(self.evaluator.evaluate("-3 + 3"), 0)

    def test_subtraction(self):
        self.assertEqual(self.evaluator.evaluate("2 - 3"), -1)
        self.assertEqual(self.evaluator.evaluate("-2 - 3"), -5)
        self.assertEqual(self.evaluator.evaluate("3 - 3"), 0)

    def test_multiplication(self):
        self.assertEqual(self.evaluator.evaluate("2 * 3"), 6)
        self.assertEqual(self.evaluator.evaluate("-2 * -2"), 4)
        self.assertEqual(self.evaluator.evaluate("-3 * 3"), -9)

    def test_division(self):
        self.assertEqual(self.evaluator.evaluate("2 / 3"), 2 / 3)
        self.assertEqual(self.evaluator.evaluate("-2 / -2"), -2 / -2)
        self.assertEqual(self.evaluator.evaluate("-3 / 3"), -3 / 3)

    def test_equality(self):
        self.assertEqual(self.evaluator.evaluate("-2 + 3 == 3 - 2"), -2 + 3 == 3 - 2)
        self.assertEqual(self.evaluator.evaluate("-2 - 3 == -3 - 2"), -2 - 3 == -3 - 2)

    def test_inequality(self):
        self.assertEqual(self.evaluator.evaluate("-2 < 3"), -2 < 3)
        self.assertEqual(self.evaluator.evaluate("-1 > -2"), -1 > -2)
        self.assertEqual(self.evaluator.evaluate("3 <= 3"), 3 <= 3)
        self.assertEqual(self.evaluator.evaluate("5 != -5"), 5 != -5)
        self.assertEqual(self.evaluator.evaluate("5 >= 3"), 5 >= 3)

    def test_logical(self):
        self.assertEqual(self.evaluator.evaluate("1 == 2 or 2 != 2"), 1 == 2 or 2 != 2)
        self.assertEqual(
            self.evaluator.evaluate("1 + 2 == 3 and not (1 + 2 == 3) or 1 // 2 == 0"),
            1 + 2 == 3 and not (1 + 2 == 3) or 1 // 2 == 0,
        )

    def test_plain_str(self):
        self.assertEqual(self.evaluator.evaluate("'afloata'"), "afloata")
        self.assertEqual(self.evaluator.evaluate("'ainta'"), "ainta")
        self.assertEqual(self.evaluator.evaluate("'aVersiona'"), "aVersiona")
        self.assertEqual(self.evaluator.evaluate("'strand'"), "strand")
        self.assertEqual(self.evaluator.evaluate("'sour'"), "sour")
        self.assertEqual(self.evaluator.evaluate("'knot'"), "knot")

    def test_function(self):
        self.assertEqual(self.evaluator.evaluate("str(123) == 123"), str(123) == 123)
        self.assertEqual(self.evaluator.evaluate("int(123) == '123'"), 123 == "123")
        self.assertEqual(
            self.evaluator.evaluate("int(str(123)) == 123"), int(str(123)) == 123
        )
        self.assertEqual(
            self.evaluator.evaluate(
                "str('$(MINDIE_USER_HOME_PATH)/Ascend/mindie-service')"
            ),
            "$(MINDIE_USER_HOME_PATH)/Ascend/mindie-service",
        )
        self.assertEqual(self.evaluator.evaluate("int('8') + int('8') // int('2')"), 12)
