class NonFiniteError(Exception):
    """Raised when a non-finite element is in the array"""
    pass


class NotInvertibleError(Exception):
    pass


class NoSolutionFoundError(Exception):
    pass
