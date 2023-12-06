import os
from qclassifier import SingleQ_classifier
from qibo import set_backend


def main():
    forma = "circle"  # or "tricrown"
    # starter = "/Users/niccolo/Desktop/tesi_magistrale/data_reuploading3/" + forma

    grid = 11
    test_samples = 4000
    method = "l-bfgs-b"

    if method == "Adam":
        set_backend("tensorflow")
    else:
        set_backend("numpy")

    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    record = True

    for layer in layers:
        folder = "Trial5_grid_" + str(test_samples) + "/"
        path = (
            "results/" + forma + "/layers_" + str(layer) + "/" + method + "/" + folder
        )

        ql = SingleQ_classifier(
            layers=layer,
            grid=grid,
            test_samples=test_samples,
            method=method,
            foldername=path,
            forma=forma,
        )

        # Guesses before training
        predicted_labels = ql.test_loop()
        ql.paint_guesses(predicted_labels, name="Trial_before_training")

        # Guesses after training
        inital_params = ql.get_parameters()
        loss, best_params = ql.training_loop()

        ql.set_parameters(best_params)
        predicted_labels = ql.test_loop()

        if record is True:
            filename = path + "Weights.txt"
            with open(filename, "a+") as file:
                file.write("\n===========================")
                name = "Trial"
                file.write(f"\n {name}")
                file.write(f"\nInitial parameters {inital_params}")
                file.write(f"\nBest parameters {best_params}")

        ql.paint_guesses(predicted_labels, name="Trial_after_training")


if __name__ == "__main__":
    main()
