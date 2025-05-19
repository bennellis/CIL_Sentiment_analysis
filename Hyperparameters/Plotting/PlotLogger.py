
"""
The idea is that we have this plot logger class that is created with the objective.
This plotlogger will gather data for each trail
at the beginning of each trail reset is called. However time intensive computations like projections will be kept allowing
for a lot faster plotting.
"""
class PlotLogger:
    def __init__(self):
        self.Embeddings = None
        self.Labels = None
        self.UncertaintyMatrix = None

    def reset(self):
        pass