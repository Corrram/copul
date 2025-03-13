import unittest
from copul.family_list import Families, FamilyCategory


class TestLazyEnum(unittest.TestCase):
    def test_list_all(self):
        """Test that list_all returns a non-empty list of family names."""
        family_names = Families.list_all()
        self.assertIsInstance(family_names, list)
        self.assertGreater(len(family_names), 0)

    def test_get_category(self):
        """Test that get_category returns the correct FamilyCategory for a given family.
        Here, we assume that 'CLAYTON' belongs to the ARCHIMEDEAN category."""
        category = Families.get_category(Families.CLAYTON)
        self.assertEqual(category, FamilyCategory.ARCHIMEDEAN)

    def test_lazy_import(self):
        """Test that accessing the 'cls' property returns a callable (the copula class)
        and that an instance of the class has a 'cdf' method (as a typical copula might).
        """
        clayton_class = Families.CLAYTON.cls
        self.assertTrue(callable(clayton_class))
        instance = clayton_class()
        self.assertTrue(hasattr(instance, "cdf"))

    def test_create(self):
        """Test that the create method returns an instance of the specified copula family.
        We check that the instance has a 'cdf' method."""
        clayton_instance = Families.create("CLAYTON")
        self.assertTrue(hasattr(clayton_instance, "cdf"))

    def test_get_params_info(self):
        """Test that get_params_info returns a dictionary for the given family."""
        params_info = Families.get_params_info("CLAYTON")
        self.assertIsInstance(params_info, dict)

    def test_list_by_category(self):
        """Test that list_by_category returns a list containing the expected family name.
        We expect 'CLAYTON' to be in the ARCHIMEDEAN category."""
        arch_families = Families.list_by_category(FamilyCategory.ARCHIMEDEAN)
        self.assertIsInstance(arch_families, list)
        self.assertIn("CLAYTON", arch_families)


if __name__ == "__main__":
    unittest.main()
