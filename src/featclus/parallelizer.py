from joblib import Parallel, delayed


class FunctionParallizer:
    """
    A class to parallelize function execution using joblib.
    """

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs

    def handle(self, func, iterable, *args, **kwargs):
        """
        Parallelizes the execution of a function over an iterable using joblib.
        The only require functionality is to provide a function and an iterable.
        The iterable must contain dataframes.

        Args:
            func (callable): The function to be parallelized. It must accept a dataframe as its first argument.
            iterable (iterable): An iterable containing dataframes to be processed.
            *args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.
        """
        return Parallel(n_jobs=self.n_jobs)(
            delayed(func)(df=item, *args, **kwargs) for item in iterable
        )
