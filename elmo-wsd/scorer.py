import os


def write_score(path_predictions, path_tests):
    """
    Calls the java scorer and writes the output in a file.
    :param path_predictions: path to predicted labels.
    :param path_tests: path to test folder.
    :return:
    """
    writer = " >> ../scores.txt"
    with open("../scores.txt", mode="w"):
        for file in path_predictions.glob("**/*.txt"):
            name, _, domain, _ = file.name.split(".")
            test_file = path_tests / name / (".".join([name, "gold", domain, "txt"]))
            os.system("echo " + str(name) + " " + domain + writer)
            os.system(
                "java -cp "
                + str(path_tests)
                + " Scorer "
                + str(file)
                + " "
                + str(test_file)
                + writer
            )
            os.system("echo " + writer)
