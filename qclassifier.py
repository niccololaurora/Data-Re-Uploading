import numpy as np
import cma
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from qibo import Circuit, gates
from dataset import create_dataset, create_target
from matplotlib.colors import Normalize


class SingleQ_classifier:

    """
    The paper uses 200 training datapoint and
    4000 test datapoint
    """

    def __init__(
        self,
        layers=1,
        grid=None,
        training_sample=200,
        test_samples=4000,
        method="l-bfgs-b",
        seed=0,
        foldername=None,
        forma="circle",
    ):
        self.epochs = 100
        self.layers = layers
        self.learning_rate = 0.001
        self.params = np.random.randn(4 * self.layers)
        self.targets = create_target(forma)
        self.training_data = create_dataset(grid=grid, seed=seed)
        self.test_data = create_dataset(samples=test_samples, seed=seed + 1)
        self.ntestsample = test_samples
        self.method = method
        self.ntrainingsample = len(self.training_data[0])
        self.foldername = foldername
        self.weights_loss = np.random.randn(2)  # il numero deve essere pari alle classi
        self.forma = forma
        if not os.path.isdir(self.foldername):
            os.makedirs(self.foldername)

    def get_parameters(self):
        return self.params

    def set_parameters(self, new_params):
        self.params = new_params

    def datapoint_circuit(self, x):
        c = Circuit(1)
        # print(f"Parametri {self.params}")
        for i in range(0, self.layers * 4, 4):
            ry = self.params[i] * x[0] + self.params[i + 1]
            rz = self.params[i + 2] * x[1] + self.params[i + 3]
            c.add(gates.RY(0, theta=ry))
            if i != (self.layers - 1) * 4:
                c.add(gates.RZ(0, theta=rz))

        return c

    def single_fidelity(self, x, y):
        c = self.datapoint_circuit(x)
        stato_finale = c.execute().state()
        cf = 0.5 * (1 - self.fidelity(stato_finale, self.targets[y])) ** 2
        return cf

    def cost_function_fidelity(self, parameters=None):
        """
        Queste tre righe servono per calcolare la loss function scegliendo i pesi.
        Se i pesi non vengono passati, di default vengono usati i pesi della classe.
        """
        if parameters is None:
            parameters = self.params
        self.set_parameters(parameters)

        cf = 0
        for x, y in zip(self.training_data[0], self.training_data[1]):
            cf += self.single_fidelity(x, y)
        cf /= len(self.training_data[0])
        return cf

    def training_loop(self, record=True):
        loss = self.cost_function_fidelity

        if self.method == "Adamax":
            print("Optimizer Adamax")

            optimizer = tf.keras.optimizers.Adamax(learning_rate=self.learning_rate)
            vparams = tf.Variable(self.params)

            @tf.function
            def opt_step():
                with tf.GradientTape() as tape:
                    l = loss(vparams)
                gradients = tape.gradient(l, [vparams])
                grads = tf.math.real(gradients)

                if record is True:
                    filename = "Gradients_" + self.method + ".txt"
                    with open(filename, "a") as file:
                        file.write("\n===========================")
                        file.write(f"\nEpoch {i+1}")
                        file.write(f"\nGradients {grads}")

                optimizer.apply_gradients(zip(grads, [vparams]))
                return l, vparams

            loss_values = []
            for i in range(self.epochs):
                print("=" * 60)
                print(f"Epoch {i+1}")
                print("=" * 60)
                l, vparams = opt_step()
                loss_values.append(l)

                if record is True:
                    filename = "Weights_" + self.method + ".txt"
                    with open(filename, "a") as file:
                        file.write("\n===========================")
                        file.write(f"\nEpoch {i+1}")
                        file.write(f"\nParameters {vparams}")

            best_loss = loss(vparams)
            self.set_parameters(vparams)
            best_parameters = vparams

        elif self.method == "Adam":
            print("Optimizer Adam")

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            vparams = tf.Variable(self.params)

            # @tf.function
            def opt_step():
                with tf.GradientTape() as tape:
                    l = loss(vparams)
                gradients = tape.gradient(l, [vparams])
                grads = tf.math.real(gradients)

                if record is True:
                    filename = "Gradients_" + self.method + ".txt"
                    with open(filename, "a") as file:
                        file.write("\n===========================")
                        file.write(f"\nEpoch {i+1}")
                        file.write(f"\nGradients {grads}")

                optimizer.apply_gradients(zip(grads, [vparams]))
                return l, vparams

            loss_values = []
            for i in range(self.epochs):
                print("=" * 60)
                print(f"Epoch {i+1}")
                print("=" * 60)
                l, vparams = opt_step()
                loss_values.append(l)

                if record is True:
                    filename = "Weights_" + self.method + ".txt"
                    with open(filename, "a") as file:
                        file.write("\n===========================")
                        file.write(f"\nEpoch {i+1}")
                        file.write(f"\nParameters {vparams}")

            best_loss = loss(vparams)
            self.set_parameters(vparams)
            best_parameters = vparams

        else:
            print("Optimizer L-BFGS-B")
            import numpy as np
            from scipy.optimize import minimize

            m = minimize(
                lambda p: loss(p),
                self.params,
                method="l-bfgs-b",
                options={"disp": True},
            )
            best_loss = m.fun
            best_parameters = m.x

        return best_loss, best_parameters

    def test_loop(self):
        labels = [[0]] * len(self.test_data[0])
        # print(f"Length dati {len(self.test_data[0])}")
        # print(f"Labels iniziali {labels}")

        for i, x in enumerate(self.test_data[0]):
            c = self.datapoint_circuit(x)
            predicted_state = c.execute().state()
            fids = np.empty(len(self.targets))

            for j, y in enumerate(self.targets):
                fids[j] = self.fidelity(predicted_state, y)

            label = np.argmax(fids)
            labels[i] = label

        print(f"Labels finali {labels}")
        return labels

    def paint_guesses(self, guess_labels, name=None):
        plt.close("all")
        x = self.test_data[0]
        x_0, x_1 = x[:, 0], x[:, 1]

        colors_rightwrong = get_cmap("viridis")
        norm_rightwrong = Normalize(vmin=-0.1, vmax=1.1)
        checks = [int(g == l) for g, l in zip(guess_labels, self.test_data[1])]
        plt.scatter(
            x_0, x_1, s=6, c=guess_labels, cmap=colors_rightwrong, norm=norm_rightwrong
        )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(
            f"Layers: {self.layers}, Method: {self.method}, Data (T, V): {self.ntrainingsample, self.ntestsample}"
        )

        if self.forma == "circle":
            center = (0, 0)
            radius = np.sqrt(2 / np.pi)
            circle = plt.Circle(center, radius, color="black", fill=False)
            plt.gca().add_patch(circle)
        else:
            center = (0, 0)
            radius = np.sqrt(0.8)
            circle1 = plt.Circle(center, radius, color="black", fill=False)
            plt.gca().add_patch(circle1)

            center = (0, 0)
            radius = np.sqrt(0.8 - 2 / np.pi)
            circle2 = plt.Circle(center, radius, color="black", fill=False)
            plt.gca().add_patch(circle2)

        print(
            "The accuracy for this classification is %.2f"
            % (100 * np.sum(checks) / len(checks)),
            "%",
        )

        if name is None:
            plt.savefig("plot.png")
        if name is not None:
            plt.savefig(self.foldername + name + ".png")

        # plt.show()

    def fidelity(self, state1, state2):
        """
        Args: two vectors
        Output: inner product of the two vectors **2
        """
        norm = tf.math.abs(tf.reduce_sum(tf.math.conj(state1) * state2))
        return norm
