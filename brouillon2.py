import numpy as np
import pandas as pd

famille_panda = [
    np.array([100, 5  , 20, 80]), # maman panda
    np.array([50 , 2.5, 10, 40]), # bébé panda
    np.array([110, 6  , 22, 80]), # papa panda
]

famille_panda_df = pd.DataFrame(famille_panda)
print(type(famille_panda_df))
