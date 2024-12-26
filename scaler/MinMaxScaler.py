from river.stats import Min, Max


class MinMaxScaler:
    """Scales the data to a fixed range from 0 to 1 with a global min max.

    Under the hood a running min and a running peak to peak (max - min) are maintained.

    Attributes
    ----------
    min : Instances of `stats.Min`.
    max : Instances of `stats.Max`.
    """

    def __init__(self):
        self.min = Min()
        self.max = Max()

    def learn_one(self, x):
        for _, xi in x.items():
            self.min.update(xi)
            self.max.update(xi)

        return self

    def transform_one(self, x):

        def safe_div(a, b):
            return a / b if b else 0.0

        return {
            i: safe_div(xi - self.min.get(), self.max.get() - self.min.get())
            for i, xi in x.items()
        }
