import numpy as np
import os
import matplotlib.pyplot as plt


file_path = os.path.join('numpy_matrix_data', 'ordered_data.npy')
loaded_array = np.load(file_path, allow_pickle=True)

print("len")
print(len(loaded_array))

print("first")
print(loaded_array[1][0])
print("second")
print(loaded_array[1][1])

plt.imshow(loaded_array[0][0])
plt.show()
