import h5py
import matplotlib.pyplot as plt
import numpy as np

f2 = h5py.File("./results10/vgg_c10_weights.h5", 'r')
f1 = h5py.File("./results9/vgg_c10_weights.h5", 'r')
f = h5py.File("./results2/vgg_c10_weights.h5", 'r')
# f = h5py.File("./results8/compressed_vgg_weights.h5", 'r')
# print(f['conv2d']['shape'][:])
# weight = f['conv2d_1']['weights'][:]
# weight=f['dense_2']['dense_2']['kernel:0'][:]
plt.figure(12, figsize=(25, 20))
plt.xlabel('weight', fontsize=40)
plt.ylabel('value', fontsize=40)
arr=[[] for i in range(0, 3)]

for key in ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'dense', 'dense_1']:
    print("key : " + key)
    weight1 = f1[key][key]['kernel:0'][:]
    weight2 = f2[key][key]['kernel:0'][:]
    weight = f[key][key]['kernel:0'][:]
    # weight = f[key]['weights'][:]
    # arr1.append([*weight1.flat])
    # arr.append([*weight.flat])
    arr[2] = arr[2] + [*weight2.flat]
    arr[1] = arr[1] + [*weight1.flat]
    arr[0] = arr[0] + [*weight.flat]
ymin = min(min(arr[0]), min(arr[1]),min(arr[2]))
ymax = max(max(arr[0]), max(arr[1]),max(arr[2]))
x = np.linspace(ymin, ymax, 60)#60
kwargs = dict(histtype='stepfilled', alpha=1, density=False, bins=x)
plt.hist(arr[1], **kwargs, label="L1")
plt.hist(arr[2], **kwargs, label="L2")
plt.hist(arr[0], **kwargs, label="VGG")

ax = plt.gca()   # 获得坐标轴的句柄

ax.spines['bottom'].set_linewidth(4)  ###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(4)    ####设置左边坐标轴的粗细
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.legend(fontsize=40)
plt.show()
