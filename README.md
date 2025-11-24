# PINTO_model_zoo-Gradio-Demo
This repository provides a Gradio demo for models from the [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo).
Because the demo code has grown large, it is published separately to avoid adding unnecessary load to issues or discussions in the original repository.

# Requirements
```
gradio
opencv-python
onnx
onnxruntime
```

# Usage
Open the Colaboratory link, run the cells from top to bottom, and access the “Running on public URL” shown in the output.

<br>
Alternatively, install the required packages locally, run the command below, and access the “Running on local URL” shown in the output.<br>

```python
python gradio_app.py
```

# Contents
| No | Model Name | Colab Link | Overview |
| --- | --- | --- | --- |
| 473 | [HISDF](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/473_HISDF) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/PINTO_model_zoo-Gradio-Demo/blob/main/473_HISDF/gradio_colab_app.ipynb) | A multitask AI model that simultaneously estimates human detection, pose, depth, and attributes. |
| 474 | [Gaze-LLE-DINOv3](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/474_Gaze-LLE-DINOv3) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/PINTO_model_zoo-Gradio-Demo/blob/main/474_Gaze-LLE-DINOv3/gradio_colab_app.ipynb) | Performs full-body detection, attribute estimation, <br>gaze heatmap generation, and skeleton <br>drawing using DEIMv2 and Gazelle. |
| 478 | [SC](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/478_SC) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/PINTO_model_zoo-Gradio-Demo/blob/main/478_SC/gradio_colab_app.ipynb) | Real-time Sitting Detection Combining Human Detection and Sitting Posture Estimation. |

# Reference
* [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo).

# License
PINTO_model_zoo-Gradio-Demo is under [Apache 2.0 license](LICENSE).
  
# Authors
高橋かずひと(https://twitter.com/KzhtTkhs)
