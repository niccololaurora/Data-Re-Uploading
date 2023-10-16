from qclassifier import SingleQ_classifier


def main():

    '''
    Eseguo il test per un solo layer 10 volte, generando i punti di training sempre in modo diverso.
    '''
    starter = "/Users/niccolo/Desktop/data_reuploading3/results/"

    training_sample = 200
    test_samples = 4000
    method = "l-bfgs-b"
    layers = 5

    for i in range(2):
        folder = "Trial_" + str(training_sample) +  "_" + str(test_samples) + "/"
        path = starter + "/layers_" + str(layers) + "/" + method + "/" + folder 
        ql = SingleQ_classifier(layers = layers, grid = None, training_sample = training_sample, test_samples = test_samples, 
        method = method, seed = i, foldername = path)
        loss, params = ql.training_loop()

        ql.set_parameters(params)
        predicted_labels = ql.test_loop()

        # print(f"Best parameters {params}")
        # print(f"Predicted labels {predicted_labels}")
        ql.paint_guesses(predicted_labels, name = "Trial_" + str(i))


if __name__ == "__main__":
    main()

