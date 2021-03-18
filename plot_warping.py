import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,15))
ax1 = plt.subplot(211)
ax1.axis('off')
ax1.set_title('Original in')
ax1.imshow(cv2.imread('test_1.jpg')[...,::-1], interpolation='nearest')

ax2 = plt.subplot(223)
ax2.axis('off')
ax2.imshow(cv2.imread('test_1_edit.jpg')[...,::-1], interpolation='nearest')
ax2.set_title('Enchanced in')

ax3 = plt.subplot(224)
ax3.axis('off')
ax3.set_title('Original warped')
ax3.imshow(img[...,::-1], interpolation='nearest')

plt.show()
