# ML-python-catboost-classifier-cpu-training


Quickstart project for executing an Wisconsin Breast Cancer classifier using the CatBoost framework on a CPU.

This Quickstart trains the model and persists as in ONNX format. The service runtime will then serve the model on localhost where the user can then send GET requests to perform inference.

* In accordance with MLOps principles, running `requirements.txt` then python `app.py` will train a model and, if threshold metrics are passed, will convert the model to `.onnx` format, saving it as `.model.onnx`.

* Additionally, metrics will be saved to a .metrics/ folder.

* Upon successful training, a Pull Request will automatically be made on the corresponding service project with the model and metrics folder being copied across.

* Jenins X requires the metrics and model to be saved in this format and the defined locations in order to promote the model to the service stage
