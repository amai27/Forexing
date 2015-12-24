import unittest

import kmeans

class KMeansTest(unittest.TestCase):


    def test_baca_file(self):
        expected = [[1.8015, 1.8015, 1.8015, 1.8015],
                    [1.8018, 1.8018, 1.8017, 1.8017],]
        results = kmeans.baca_file('tests/test_data.csv')
        self.assertTrue(expected, results)

