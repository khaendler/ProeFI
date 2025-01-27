from river.datasets.synth import Agrawal
import time

from tree import HoeffdingPruningTree, HTMerit, HPTMerit, HPTConvexMerit
from river.tree import ExtremelyFastDecisionTreeClassifier


stream = Agrawal(classification_function=1, seed=42)
model1 = HPTConvexMerit()
model2 = HTMerit()
feature_names = ["salary", "commission", "age", "elevel", "car", "zipcode", "hvalue", "hyears", "loan"]
avg_time1 = 0
avg_time2 = 0
for i, (x, y) in enumerate(stream.take(100000), start=1):
    start = time.perf_counter()
    model1.learn_one(x, y)
    avg_time1 += time.perf_counter() - start
    start = time.perf_counter()
    model2.learn_one(x, y)
    avg_time2 += time.perf_counter() - start

print(avg_time1 / 100000)
print(avg_time2 / 100000)

#highlighted_keys=["salary", "commission", "age", "elevel"]
#.plot_pfi(names_to_highlight=highlighted_keys)
#model2.plot_feature_importance(names_to_highlight=highlighted_keys)

