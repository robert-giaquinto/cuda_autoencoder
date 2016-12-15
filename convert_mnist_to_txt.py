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
        ol.write(" ".join([str(label) for label in labels]))

    with open(out_trainf, "w") as ot:
        for image in images:
            ot.write(" ".join(["%.4f" % (float(pix)/255.0) for pix in image]))

    f.close()

convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "mnist_train.txt", "mnist_train_labels.txt", 60000)
convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "mnist_test.txt", "mnist_test_labels.txt", 10000)
