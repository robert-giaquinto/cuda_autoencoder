def convert(imgf, labelf, out_trainf, out_labelf, n):
    f = open(imgf, "rb")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []
    labels = []

    for i in range(n):
        labels.append(ord(l.read(1)))
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    with open(out_labelf, "w") as ol:
        for label in labels:
            ol.write(str(label) + '\n')

    with open(out_trainf, "w") as ot:
        for image in images:
            ot.write(",".join(str(float(pix)/255.0) for pix in image) + "\n")

    f.close()

convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "mnist_train.csv", "mnist_train_labels.csv", 60000)
convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "mnist_test.csv", "mnist_test_labels.csv", 10000)
